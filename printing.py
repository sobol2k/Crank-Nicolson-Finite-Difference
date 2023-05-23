from os import path
import matplotlib.pyplot as plt

def print_solution(solution,filepath:str,filename:str):
    
    filepath_complete = path.join(filepath, filename)

    # Create heatmap of solution
    fig = plt.figure()
    plt.imshow(solution["sol"], cmap='viridis', interpolation='nearest')
    plt.xlabel("x")
    plt.ylabel("t")
    fig.savefig(filepath_complete, dpi=300)
