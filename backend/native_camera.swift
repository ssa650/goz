// UID-based AVFoundation capture. `list` never creates an input or starts a session.
import AVFoundation
import Foundation
import CoreVideo

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func devices() -> [AVCaptureDevice] {
    var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera]
    if #available(macOS 14.0, *) {
        types += [.external, .continuityCamera]
    } else {
        types += [.externalUnknown]
    }
    return AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video,
                                            position: .unspecified).devices
}

func identity(_ device: AVCaptureDevice) -> [String: Any] {
    return ["deviceId": device.uniqueID, "name": device.localizedName,
            "backend": "avfoundation-uid"]
}

func jsonLine(_ value: Any) {
    do {
        var data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        data.append(10)
        FileHandle.standardOutput.write(data)
    } catch { fail("Could not encode camera identity: \(error)") }
}

final class Frames: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sample: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sample) else { return }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var data = Data()
        // Packed BGRA rows; fixed 12-byte little-endian header.
        for value in [UInt32(width), UInt32(height), UInt32(width * height * 4)] {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }
        for row in 0..<height {
            data.append(base.advanced(by: row * stride).assumingMemoryBound(to: UInt8.self),
                        count: width * 4)
        }
        FileHandle.standardOutput.write(data)
    }
}

let args = CommandLine.arguments
guard args.count >= 2 else { fail("Use list or capture <device UID>.") }
if args[1] == "list" {
    jsonLine(devices().enumerated().map { index, device in
        var info = identity(device)
        info["index"] = index  // UI ordinal only, never an OpenCV capture index.
        return info
    })
    exit(0)
}
guard args.count == 3 && args[1] == "capture" else { fail("Use list or capture <device UID>.") }
let uid = args[2]
guard let selected = devices().first(where: { $0.uniqueID == uid }) else {
    fail("Selected camera is unavailable. Reconnect it and refresh the selector; no other camera was opened.")
}
switch AVCaptureDevice.authorizationStatus(for: .video) {
case .authorized: break
case .notDetermined:
    let permission = DispatchSemaphore(value: 0)
    var allowed = false
    AVCaptureDevice.requestAccess(for: .video) { result in allowed = result; permission.signal() }
    permission.wait()
    if !allowed { fail("Camera access was denied. Check macOS Privacy & Security > Camera, then retry.") }
default: fail("Camera access is unavailable. Check macOS Privacy & Security > Camera, then retry.")
}

let session = AVCaptureSession()
let input: AVCaptureDeviceInput
do { input = try AVCaptureDeviceInput(device: selected) }
catch { fail("Could not open selected camera \(selected.localizedName): \(error)") }
guard input.device.uniqueID == uid else { fail("Opened camera UID differs from selection. Full recalibration is required.") }
session.beginConfiguration()
if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 }
guard session.canAddInput(input) else { fail("Selected camera cannot be added to capture; no fallback was attempted.") }
session.addInput(input)
let output = AVCaptureVideoDataOutput()
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
output.alwaysDiscardsLateVideoFrames = true
guard session.canAddOutput(output) else { fail("Selected camera cannot deliver video frames.") }
session.addOutput(output)
if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
    connection.automaticallyAdjustsVideoMirroring = false
    connection.isVideoMirrored = false  // Gazekit applies its own single mirror.
}
let frames = Frames()
let queue = DispatchQueue(label: "goz.camera.frames")
queue.suspend() // Identity handshake must precede all frame packets.
output.setSampleBufferDelegate(frames, queue: queue)
session.commitConfiguration()
session.startRunning()
guard session.isRunning else { fail("Selected camera failed to start. Reconnect it and retry.") }
var opened = identity(input.device)
opened["verified"] = true
jsonLine(opened)
queue.resume()
// Closing the parent's stdin ends only this owned capture session.
_ = FileHandle.standardInput.readDataToEndOfFile()
session.stopRunning()
