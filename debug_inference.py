import faulthandler
import sys
import multiprocessing
from inference import main as inference_main

# Enable faulthandler to get tracebacks on hard crashes or hangs
faulthandler.enable()

def wrapper_main():
    """
    This wrapper runs the original inference.py main function in a separate
    process. This helps in debugging hangs by allowing a timeout mechanism,
    and faulthandler can provide tracebacks for critical errors.
    """
    # We don't need a queue for this simple case, as the child process
    # will inherit the stdout/stderr streams. The main goal is isolation
    # and getting a traceback from faulthandler if the process hangs.
    # The `app.py` will handle the timeout at the subprocess.run level.

    print("--- Starting inference via debug_inference.py wrapper ---")

    try:
        inference_main()
    except Exception as e:
        print(f"--- Debug wrapper caught an exception: {e} ---", file=sys.stderr)
        # Re-raise the exception to ensure it propagates
        raise

if __name__ == '__main__':
    # The command line arguments are already in sys.argv, and inference_main
    # uses argparse to parse them directly from there. So, we just need to call it.
    wrapper_main()
