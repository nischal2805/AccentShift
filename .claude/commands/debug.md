# Debug a pipeline error

Debug the error below. 

Process:
1. Identify the root cause — don't just fix the symptom
2. Check if it violates any constraint in CLAUDE.md (dtype, sample rate, segment length, VRAM)
3. Show the minimal fix with explanation
4. If it's a CUDA OOM, suggest which model to move to CPU first (priority: Vevo → SER → Whisper)

Error and context: $ARGUMENTS
