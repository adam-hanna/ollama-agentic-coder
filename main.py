#!/usr/bin/env python3

import tracemalloc

# Enable tracemalloc to provide better debugging info for memory issues
tracemalloc.start()

from cli.main import main

if __name__ == "__main__":
    main()
