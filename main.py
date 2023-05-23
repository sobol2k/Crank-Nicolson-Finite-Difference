from PDE import *
from printing import *

""" 
    Implements the diffusion reaction PDE with a convection term in one dimension 
    using the Crank Nicholson schema while using the upwind schema for the  v_x*u_x term.

    u_t(x,t) - D * u_xx(x,t) + v_x*u_x(x,t) + alpha u(x,t) = q(x,t)

    within x in (0,L). We assume q(x,t) = q_0(t) exp(-(x-x_0)^2/2*sigma_x^2) 
    as the source term.

    Initial condition   : c(x,0) = c_0(x) = 0 
    Boundary conditions : 
    (i)  Dirichlet boundary conditions at c(0,t) = 0 \forall t
    (ii) Neumann boundary conditions at c(L,t) = ||v|| u(x,t)

"""

def main():
        
    # Create grid
    start           = 0
    end             = 10
    number_of_spatial_nodes = 500
    spatial_grid = create_grid(start,end,number_of_spatial_nodes)

    # Create time stepping
    start           = 0
    end             = 10
    number_of_nodes = 500
    time_grid = create_grid(start,end,number_of_nodes)

    # Initial values
    y0 = np.zeros(number_of_spatial_nodes)

    # Parameters
    problem_parameters = {"diffusion_coefficient":0.001,
                          "velocity_x":1.0,
                          "alpha":0.001,}
    
    boundary_parameters = {"dirichlet":0.0}
    
    # Numerical weighting
    weighting_factor = 0.25
    
    # Solve problem
    solution = solve_problem(y0,spatial_grid,time_grid,weighting_factor,
                             problem_parameters,boundary_parameters,source_term,
                             verbose = True)
    


    # Print solution
    filepath = "F:/Uni/Zuse/simlopt/simlopt/"
    print_solution(solution,filepath,"heatmap.png")

if __name__ == "__main__":
    main()
