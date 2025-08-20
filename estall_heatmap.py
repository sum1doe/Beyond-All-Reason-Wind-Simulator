import matplotlib.pyplot as plt
from wind_simulator import Simulator

# Generate Heatmap for estall duration

def run_sim(verbose = False):
    array_2d = [[0 for _ in range(21)] for _ in range(10)]        

    for min_wind in range(0, 10):
        for max_wind in range(10,21):
            print(f"Reached ({min_wind}, {max_wind})/({9}, {20})") if verbose else None
            sim = Simulator()
            res = sim.simulate_wind_drop(1000, min_wind,  max_wind)
            array_2d[min_wind][max_wind] = res
            
    return array_2d
            
def plot(result):
    plt.imshow(result)
    plt.xlim(10,20)
    plt.colorbar() 
    plt.show()
    
if __name__ == "__main__":
    plot(run_sim())