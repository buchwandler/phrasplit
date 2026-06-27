# Streaming API

## Overview

The streaming API provides memory-efficient, incremental processing of text segments.
Instead of loading all segments into memory at once, the iterator yields segments one by
one in document order.

This is particularly useful for:

- **Large documents**: Process multi-gigabyte texts without loading all segments into
  memory
- **Real-time synthesis**: Start TTS processing before all segmentation is complete
- **Pipeline integration**: Stream segments through processing stages
- **Low-latency applications**: Begin output as soon as first segment is ready

## Iterator Function

```python
from phrasplit import iter_split_with_offsets

for segment in iter_split_with_offsets(text, mode="sentence"):
    print(f"{segment.id}: {segment.text}")
```

The iterator signature matches `split_with_offsets()`:

```python
iter_split_with_offsets(
    text: str,
    *,
    mode: str = "sentence",
    use_spacy: bool | None = None,
    language_model: str = "en_core_web_sm",
    max_chars: int | None = None,
) -> Iterator[SplitSegment]
```

## Guarantees

### Document Order

Segments are always yielded in document order (by `char_start`):

```python
segments = []
for seg in iter_split_with_offsets(text):
    if segments:
        # Current segment always comes after previous
        assert seg.char_start >= segments[-1].char_start
    segments.append(seg)
```

### No Global State

Each iterator is independent with no shared state:

```python
# Safe to run multiple iterators
iter1 = iter_split_with_offsets(text1)
iter2 = iter_split_with_offsets(text2)

# No interference between iterations
seg1 = next(iter1)
seg2 = next(iter2)
```

### Same Offsets

Offset guarantees are identical to `split_with_offsets()`:

```python
for segment in iter_split_with_offsets(text):
    # Offsets map exactly to original text
    assert text[segment.char_start:segment.char_end] == segment.text
```

## Usage Examples

### Basic Streaming

```python
from phrasplit import iter_split_with_offsets

text = "First sentence. Second sentence.\\n\\nNew paragraph."

for segment in iter_split_with_offsets(text, mode="sentence"):
    print(f"Processing {segment.id}: {segment.text}")
    # Process immediately without waiting for all segments
```

### Real-time TTS Pipeline

```python
import asyncio
from phrasplit import iter_split_with_offsets

async def synthesize_segment(segment):
    """Synthesize audio for one segment."""
    # Hypothetical TTS call
    audio = await tts_engine.synthesize(segment.text)
    return {
        "segment_id": segment.id,
        "audio": audio,
        "position": (segment.char_start, segment.char_end)
    }

async def stream_synthesis(text):
    """Stream synthesis results as soon as each segment is ready."""
    for segment in iter_split_with_offsets(text, mode="sentence"):
        result = await synthesize_segment(segment)
        yield result

# Usage
async for audio_chunk in stream_synthesis(long_text):
    await audio_output.write(audio_chunk)
```

### Processing Large Files

```python
def process_large_book(file_path, output_path):
    """Process a large ebook segment by segment."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    with open(output_path, 'w', encoding='utf-8') as out:
        for segment in iter_split_with_offsets(text, mode="sentence"):
            # Process one segment at a time
            processed = process_segment(segment)

            # Write immediately
            out.write(f"{segment.id}\\t{processed}\\n")

            # Memory is freed after each iteration
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
from phrasplit import iter_split_with_offsets

def process_batch(segments):
    """Process a batch of segments."""
    results = []
    for seg in segments:
        result = expensive_operation(seg.text)
        results.append((seg.id, result))
    return results

def stream_with_batching(text, batch_size=10):
    """Stream segments in batches for parallel processing."""
    batch = []

    for segment in iter_split_with_offsets(text, mode="sentence"):
        batch.append(segment)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield remaining
    if batch:
        yield batch

# Process batches in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    for batch in stream_with_batching(long_text, batch_size=20):
        future = executor.submit(process_batch, batch)
        results = future.result()
        # Handle results...
```

### Progress Tracking

```python
from phrasplit import iter_split_with_offsets

def process_with_progress(text):
    """Process with progress tracking."""
    total_chars = len(text)

    for segment in iter_split_with_offsets(text, mode="sentence"):
        # Calculate progress based on character position
        progress = (segment.char_end / total_chars) * 100

        print(f"Progress: {progress:.1f}% - Processing {segment.id}")

        # Process segment...
        process_segment(segment)
```

### Selective Processing

```python
def process_matching_segments(text, pattern):
    """Process only segments matching a pattern."""
    import re

    for segment in iter_split_with_offsets(text, mode="sentence"):
        if re.search(pattern, segment.text):
            # Only process matching segments
            result = expensive_processing(segment)
            yield (segment.id, result)
        else:
            # Skip non-matching segments
            yield (segment.id, None)
```

### Early Termination

```python
def find_first_mention(text, search_term):
    """Find first segment containing search term."""
    for segment in iter_split_with_offsets(text, mode="sentence"):
        if search_term in segment.text:
            # Found it - stop iteration
            return segment

    return None

# Iterator stops as soon as we return
result = find_first_mention(long_document, "important phrase")
```

## Comparison with List Version

### When to Use Iterator

✅ Use `iter_split_with_offsets()` when:

- Processing very large texts (> 1 MB)
- Need to start output before all segmentation is done
- Building real-time pipelines
- Memory is constrained
- Can process segments independently

### When to Use List

✅ Use `split_with_offsets()` when:

- Need random access to segments
- Need total segment count upfront
- Performing multiple passes over segments
- Text size is small (< 1 MB)
- Need to sort or filter all segments

## Performance Characteristics

### Memory Usage

```python
# List version - loads all segments into memory
segments = split_with_offsets(huge_text)  # Uses O(n) memory

# Iterator version - one segment at a time
for seg in iter_split_with_offsets(huge_text):  # Uses O(1) memory
    process(seg)
```

### Time to First Segment

The iterator has the same time-to-first-segment as the list version since the current
implementation processes all segments upfront. A future optimization could make this
truly streaming.

```{note}
Current implementation note: The iterator currently uses `split_with_offsets()` internally and yields from the result. A future version may implement true streaming for faster time-to-first-segment.
```

### Overhead

Iterator overhead is minimal:

```python
# List conversion
segments_list = list(iter_split_with_offsets(text))

# Equivalent to
segments_list = split_with_offsets(text)
```

## Best Practices

1. **Process Immediately**

   ```python
   # Good - process each segment as it arrives
   for seg in iter_split_with_offsets(text):
       result = process(seg)
       save(result)

   # Bad - defeats purpose of streaming
   all_segments = list(iter_split_with_offsets(text))
   ```

2. **Use with Context Managers**

   ```python
   with open('output.txt', 'w') as f:
       for seg in iter_split_with_offsets(text, mode="sentence"):
           processed = process(seg)
           f.write(f"{processed}\\n")
   ```

3. **Combine with Generators**

   ```python
   def processed_segments(text):
       """Generator pipeline."""
       for seg in iter_split_with_offsets(text):
           if should_process(seg):
               yield process(seg)

   # Chain generators
   for result in processed_segments(text):
       output(result)
   ```

## See Also

- {doc}`offsets` - Offset coordinate system and guarantees
- {doc}`api` - Complete API reference
- {doc}`examples` - More usage examples
