import numpy as np
import time
import logging
import matplotlib.pyplot as plt

from dataclasses import dataclass
from functools import wraps
from scipy.sparse import diags

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def create_grid(start,end,number_of_nodes):
    # Number_of_nodes includes begin and start node
    # Number of inner nodes:  number_of_nodes-2
    # Number of intervals:    number_of_nodes-1
    if end <= start:
        raise ValueError('End point cant before or the start point')
    h = (end - start)/ (number_of_nodes-1) 
    grid_data = {"grid_values": np.linspace(start,end,number_of_nodes),
                 "grid_spacing": h}
    return grid_data


def create_system_matrx(time_step,node_spacing,num_nodes,parameters,lhs = True):
    

    """
    
    At the moment \theta = 0.5 is implemented. Will be changed for more general implementaion with 0 \leq \theta \leq 1
    """

    # Problem parameters
    diffusion_coefficient = parameters.get("diffusion_coefficient",1)
    velocity_x            = parameters.get("velocity_x",0.1)
    alpha                 = parameters.get("alpha",0.1)

    # Validate input
    if not isinstance(time_step, float):
        logger.error("Time step must be a float.")
        raise TypeError("Time step must be a float.")
    if not isinstance(diffusion_coefficient, float):
        logger.error("Diffusion coefficient must be a float.")
        raise TypeError("Diffusion coefficient must be a float.")
    if not isinstance(node_spacing, float):
        logger.error("Node spacing must be a float.")
        raise TypeError("Node spacing must be a float.")
    if not isinstance(velocity_x, float):
        logger.error("Velocity in the x-direction must be a float.")
        raise TypeError("Velocity in the x-direction must be a float.")
    if not isinstance(num_nodes, int):
        logger.error("Number of nodes must be an integer.")
        raise TypeError("Number of nodes must be an integer.")
    
    coefficient  = diffusion_coefficient / (4*node_spacing**2)

    left_coefficient  = (-time_step*(coefficient  + np.maximum(velocity_x,0)/ 2*node_spacing))
    right_coefficient = (-time_step*(coefficient  - np.minimum(velocity_x,0)/ 2*node_spacing))

    central_vector  = (time_step/2*(diffusion_coefficient  / node_spacing**2) + time_step*np.abs(velocity_x) / 2*node_spacing + time_step*alpha/2)*np.ones(num_nodes)
    left_vector     = left_coefficient*np.ones(num_nodes-1)
    right_vector    = right_coefficient*np.ones(num_nodes-1)

    unit_matrix     = np.eye(num_nodes)

    k = [left_vector,central_vector,right_vector]
    offset = [-1,0,1]
    LHS_system_matrx = diags(k,offset).toarray()

    if lhs:
        LHS_system_matrx =  LHS_system_matrx+unit_matrix
    else:
        LHS_system_matrx = -LHS_system_matrx+unit_matrix

    # Set Neumann boundary condition
    neumann_coeffcient = (2*node_spacing*right_coefficient) / diffusion_coefficient *(np.linalg.norm(velocity_x)+np.abs(velocity_x))
    LHS_system_matrx[-1,-1] += neumann_coeffcient
    LHS_system_matrx[-1,-2] += right_coefficient
    
    #TODO: Variabel über theta

    return LHS_system_matrx


def source_term(x,t,x0,sigma):
    q0   = 0.01*t
    return q0 * np.exp( -(x0-x)**2 / 2*sigma**2 ) 

def solve_problem(y0,spatial_grid,time_grid,parameters:dict,source_term,verbose=False):

    # Spatial grid data
    spatial_values = spatial_grid.get("grid_values")
    h_spatial      = spatial_grid.get("grid_spacing")
    num_nodes      = spatial_values.shape[0]

    # Check is y0 has same size as size of inner nodes
    try:
        y0.shape[0] == num_nodes
    except:
        raise TypeError("Time step must be a float.")

    # Time grid data
    time_values = time_grid.get("grid_values")
    h_time      = time_grid.get("grid_spacing")

    # Timing
    t_total = 0 

    # Source parameters:
    x0 = np.array([0.5])
    sigma = 10

    # Solve equation system for all time steps
    solution_array = np.zeros((time_values.shape[0],
                               spatial_values.shape[0]))
    
    # Set Dirichlet bounday conditions and save start values
    #TODO:Boundary values via bounday dict
    y0[0] = 0
    solution_array[0,:] = y0

    for t_iter in range(0,time_values.shape[0]-1):

        # Current solution
        y = solution_array[t_iter,:]     

        # Current time
        current_time = time_values[t_iter]
        next_time    = time_values[t_iter+1]

        # Create system matrix LHS
        system_matrix_lhs = create_system_matrx(h_time,h_spatial,num_nodes,parameters,lhs = True)
        
        # Create RHS
        # The last addition of source_term is due to Crank Nicholson schema.
        system_matrix_rhs = create_system_matrx(h_time,h_spatial,num_nodes,parameters,lhs = False)
        rhs               = system_matrix_rhs@y
        rhs               += h_time / 2 * source_term(spatial_values,current_time,x0,sigma)
        rhs               += h_time / 2 * source_term(spatial_values,next_time,x0,sigma)

        if verbose:
            logger.info("Iteration:                     %i  ", t_iter )    
            logger.info("Current time step:             %f  ", time_values[t_iter])    
            #logger.info("Time creating system matrix:   %.4f", t_create_LHS_total )
            #logger.info("Time creating right hand side: %.4f", t_create_RHS_total )
            logger.info("Total time creating problem:   %.4f", t_total )

        # Solve LGS
        solution = np.linalg.solve(system_matrix_lhs,rhs)

        # Set Dirichlet bounday conditions
        solution[0] = 0.0
            
        # Save solution
        solution_array[t_iter+1,:] = solution
        y0 = solution
                
    solution_array[-1,:] = solution
    
    return {"sol":solution_array}


def main():
        
    # Create grid
    start           = 0
    end             = 1
    number_of_spatial_nodes = 200
    spatial_grid = create_grid(start,end,number_of_spatial_nodes)

    # Create time stepping
    start           = 0
    end             = 10
    number_of_nodes = 200
    time_grid = create_grid(start,end,number_of_nodes)

    # Initial values
    y0 = np.zeros(number_of_spatial_nodes)

    # Parameters
    parameters = {"diffusion_coefficient":0.0001,
                  "velocity_x":1.0,
                  "alpha":0.001,}

    # Solve problem
    solution = solve_problem(y0,spatial_grid,time_grid,parameters,source_term,verbose = True)
    
    # Print solution
    plt.imshow(solution["sol"], cmap='hot', interpolation='nearest')
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

if __name__ == "__main__":
    main()
