"""Simple face detection script using Google Cloud Vision.

Supports single-image detection and directory/batch processing.

Usage:
  python facialrec.py /path/to/image.jpg            # single image
  python facialrec.py --dir /path/to/images         # detect all images in directory in batches

Notes:
  - Requires the `google-cloud-vision` package and valid GCP credentials
    (set GOOGLE_APPLICATION_CREDENTIALS or use ADC).
"""

import io
import os
import sys
import argparse
from typing import List, Iterable, Any
try:
    from google.cloud import vision
    HAS_VISION = True
except Exception:  # pragma: no cover - optional dependency
    vision = None
    HAS_VISION = False


MAX_BATCH = 16  # Vision API max images per batch request


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    """Yield successive chunks from iterable of given size."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def gather_images_from_dir(directory: str) -> List[str]:
    """Return a sorted list of image file paths from directory.

    Accepts common extensions .jpg, .jpeg, .png
    """
    exts = ('.jpg', '.jpeg', '.png')
    files = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.lower().endswith(exts)]
    return files


def detect_faces(image_path: str, client: Any = None, dry_run: bool = False):
    """Detect faces in a single image and return the response object.

    Returns tuple (image_path, response) where response is the API response.
    """
    close_client = False
    if dry_run or not HAS_VISION:
        # Dry-run mode: don't call the API, just report we would process the file
        print(f"[dry-run] Would process: {image_path}")
        return image_path, None

    if client is None:
        client = vision.ImageAnnotatorClient()
        close_client = True

    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
    except FileNotFoundError:
        raise

    image = vision.Image(content=content)
    response = client.face_detection(image=image)

    if close_client:
        # client has no close() method on older libs, leave to GC
        pass

    return image_path, response


def detect_faces_batch(image_paths: List[str], dry_run: bool = False):
    """Process images in batches (up to MAX_BATCH) using the Vision API.

    Prints results for each image and returns 0 on success. On API error prints and returns non-zero.
    """
    if dry_run or not HAS_VISION:
        for batch_num, batch in enumerate(chunked(image_paths, MAX_BATCH), start=1):
            print(f"[dry-run] Batch {batch_num}: would send {len(batch)} image(s) to the API")
            for image_path in batch:
                print(f"  [dry-run] {image_path}")
        return 0

    client = vision.ImageAnnotatorClient()

    for batch_num, batch in enumerate(chunked(image_paths, MAX_BATCH), start=1):
        requests = []
        for image_path in batch:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            requests.append({'image': image})

        # The client has a batch_annotate_images method that accepts a list of requests.
        # Each request can specify which features to enable; here we only need FACE_DETECTION.
        feature = vision.Feature(type_=vision.Feature.Type.FACE_DETECTION)
        api_requests = [{'image': req['image'], 'features': [feature]} for req in requests]

        response = client.batch_annotate_images(requests=api_requests)

        # response.responses is a list aligned to the input requests
        for image_path, res in zip(batch, response.responses):
            if res.error.message:
                print(f"API error for '{image_path}': {res.error.message}")
                return 2

            faces = res.face_annotations
            print(f"Batch {batch_num}: Detected {len(faces)} face(s) in '{image_path}'")
            for i, face in enumerate(faces, start=1):
                dc = getattr(face, 'detection_confidence', None)
                print(f"  Face #{i}: detection_confidence={dc:.2f}" if dc is not None else f"  Face #{i}")
                try:
                    joy = vision.Likelihood(face.joy_likelihood).name
                    sorrow = vision.Likelihood(face.sorrow_likelihood).name
                    anger = vision.Likelihood(face.anger_likelihood).name
                    surprise = vision.Likelihood(face.surprise_likelihood).name
                    print(f"    joy={joy}, sorrow={sorrow}, anger={anger}, surprise={surprise}")
                except Exception:
                    print(f"    joy={face.joy_likelihood}, sorrow={face.sorrow_likelihood}, anger={face.anger_likelihood}, surprise={face.surprise_likelihood}")

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Face detection using Google Cloud Vision')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', '-i', help='Path to a single image')
    group.add_argument('--dir', '-d', help='Path to a directory containing images')
    parser.add_argument('--dry-run', action='store_true', help='Do not call the API; simulate processing')

    args = parser.parse_args(argv)

    if args.image:
        try:
            _, res = detect_faces(args.image, dry_run=args.dry_run)
        except FileNotFoundError:
            print(f"File not found: {args.image}")
            return 1

        if not args.dry_run:
            if res.error.message:
                print(f"API error: {res.error.message}")
                return 2

            faces = res.face_annotations
            print(f"Detected {len(faces)} face(s) in '{args.image}'")
            for i, face in enumerate(faces, start=1):
                dc = getattr(face, 'detection_confidence', None)
                print(f"Face #{i}: detection_confidence={dc:.2f}" if dc is not None else f"Face #{i}")

        return 0

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Not a directory: {args.dir}")
            return 1
        image_paths = gather_images_from_dir(args.dir)
        if not image_paths:
            print(f"No images found in {args.dir}")
            return 1
        return detect_faces_batch(image_paths, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())