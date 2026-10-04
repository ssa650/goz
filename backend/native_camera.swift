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
    // One pending frame plus one pipe write in flight. Never block AVFoundation
    // on the pipe or enqueue an unbounded history of video buffers.
    private let lock = NSLock()
    private let available = DispatchSemaphore(value: 0)
    private var pending: (Data, [String: Any])?
    private var sequence: UInt64 = 0
    private var delivered: UInt64 = 0
    private var dropped: UInt64 = 0
    private var replaced: UInt64 = 0
    private var writeMs: Double = 0
    private var lastCallbackAt: Double = 0
    private var writingSince: Double = 0

    override init() {
        super.init()
        DispatchQueue(label: "goz.camera.pipe").async { self.writeFrames() }
    }

    private func writeFrames() {
        while true {
            available.wait()
            lock.lock()
            guard let (pixels, values) = pending else { lock.unlock(); continue }
            pending = nil
            var metadata = values
            metadata["nativeDelivered"] = delivered
            metadata["nativeDropped"] = dropped
            metadata["nativeReplaced"] = replaced
            metadata["previousWriteMs"] = writeMs
            writingSince = ProcessInfo.processInfo.systemUptime
            lock.unlock()
            guard let json = try? JSONSerialization.data(withJSONObject: metadata) else { continue }
            var packet = Data()
            var length = UInt32(json.count).littleEndian
            withUnsafeBytes(of: &length) { packet.append(contentsOf: $0) }
            packet.append(json)
            packet.append(pixels)
            let start = ProcessInfo.processInfo.systemUptime
            FileHandle.standardOutput.write(packet)
            lock.lock()
            writeMs = (ProcessInfo.processInfo.systemUptime - start) * 1000
            writingSince = 0
            lock.unlock()
        }
    }

    func diagnostics() -> [String: Any] {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock(); defer { lock.unlock() }
        return ["nativeDelivered": delivered, "nativeDropped": dropped,
            "nativeReplaced": replaced, "nativePending": pending == nil ? 0 : 1,
            "nativeCallbackAgeS": lastCallbackAt == 0 ? -1 : now-lastCallbackAt,
            "nativeWriteAgeS": writingSince == 0 ? 0 : now-writingSince,
            "previousWriteMs": writeMs, "nativeHeartbeatAt": Date().timeIntervalSince1970]
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sample: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        lock.lock(); dropped += 1; lastCallbackAt = ProcessInfo.processInfo.systemUptime; lock.unlock()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sample: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let deliveredMono = ProcessInfo.processInfo.systemUptime
        lock.lock(); lastCallbackAt = deliveredMono; lock.unlock()
        let deliveredAt = Date().timeIntervalSince1970
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
        let hostNow = CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
        // AVFoundation video PTS is on the host clock. Preserve acquisition
        // time rather than retimestamping an old pipe frame as fresh.
        let age = hostNow - pts
        guard age.isFinite && age >= -0.1 else { return }
        guard let buffer = CMSampleBufferGetImageBuffer(sample) else { return }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var data = Data()
        for row in 0..<height {
            data.append(base.advanced(by: row * stride).assumingMemoryBound(to: UInt8.self),
                        count: width * 4)
        }
        lock.lock()
        sequence += 1; delivered += 1
        let signal = pending == nil
        if !signal { replaced += 1 }
        pending = (data, ["width": width, "height": height, "length": data.count,
            "sequence": sequence, "nativePTS": pts,
            "capturedAt": deliveredAt - age, "capturedMonotonic": deliveredMono - age,
            "deliveredAt": deliveredAt, "deliveredMonotonic": deliveredMono])
        lock.unlock()
        if signal { available.signal() }
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
opened["frameProtocol"] = 2
jsonLine(opened)
queue.resume()
// Separate low-volume diagnostic channel continues even if the pixel pipe is
// blocked. It distinguishes a live callback from capture silence in a stall.
let heartbeat = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "goz.camera.diagnostics"))
heartbeat.schedule(deadline: .now(), repeating: .seconds(2))
heartbeat.setEventHandler {
    if let data = try? JSONSerialization.data(withJSONObject: frames.diagnostics()) {
        FileHandle.standardError.write(Data("GOZ_NATIVE_DIAGNOSTICS ".utf8) + data + Data([10]))
    }
}
heartbeat.resume()
// Closing the parent's stdin ends only this owned capture session.
_ = FileHandle.standardInput.readDataToEndOfFile()
heartbeat.cancel()
session.stopRunning()
