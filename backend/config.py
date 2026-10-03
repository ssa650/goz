"""Non-secret model configuration, ported from GOZ_TEST/config.js."""
MODELS = {
    "text": "minimax/h3-max-turbo/text-to-video",
    "frames": "minimax/h3-max-turbo/image-to-video",
    "characters": "minimax/h3-max/reference-to-video",
    "combined": "minimax/h3-max/reference-to-video",
}
DURATION = 15
DURATIONS = list(range(5, 16))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_CLIPS = 12
MAX_CHARACTERS = 7
MAX_PROMPT_BYTES = 100_000
POLL_SECONDS = 0.5


def build_input(options, urls):
    mode = options["mode"]
    duration = options.get("duration", DURATION)
    result = dict(prompt=options["prompt"], duration=duration,
                  resolution=options.get("resolution", "480P"),
                  prompt_expansion_mode=options.get("promptExpansionMode", "disabled"), enable_safety_checker=True, sync_mode=False)
    if options.get("seed") is not None:
        result["seed"] = options["seed"]
    if mode == "text":
        result["aspect_ratio"] = options.get("aspectRatio", "16:9")
        return result
    if mode == "frames":
        result["image_url"] = urls["start"]
        if urls.get("end"):
            result["end_image_url"] = urls["end"]
        return result
    result["aspect_ratio"] = options.get("aspectRatio", "16:9")
    references, instructions = [], []
    if mode == "combined":
        references.extend([urls["start"], urls["end"]])
        instructions.append(f"Image 1 is the desired opening composition. Begin the video with a scene matching Image 1. Image 2 is the desired closing composition. End the video with a scene matching Image 2. Transition naturally between these scenes over {duration} seconds.")
    names = options.get("names", [])
    for i, url in enumerate(urls.get("characters", [])):
        references.append(url)
        name = names[i] if i < len(names) and names[i] else f"Character {i + 1}"
        instructions.append(f"Image {len(references)} is the character reference for {name}. Keep this character consistent with that image throughout the video.")
    result["reference_image_urls"] = references
    result["prompt"] = "\n".join(instructions) + "\n\nUser direction:\n" + options["prompt"]
    return result
