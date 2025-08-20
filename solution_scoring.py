import cProfile
import pstats
import io

import estall_heatmap

if __name__ == "__main__":
    # Create and start the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the code we want to profile
    result = estall_heatmap.run_sim(verbose=True)
    print(f"Estall_heatmap has finished running.")

    # Disable the profiler and print stats
    profiler.disable()
    estall_heatmap.plot(result)

    # Format and display the results
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
