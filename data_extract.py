import pandas as pd
import openpyxl as xl
import pickle

"""
Organize all data about desired performance error and actual performance error as an Excel file. Ready for MATLAB plot
display
"""

extra_disc_para = 5
epsilon_list = [0.5, 0.3, 0.2]
tot_cost_list = []
perform_bound_list = []

for epsilon in epsilon_list:
    sim_file = open(f'sim_result_new (epsilon = {epsilon}, extra_disc_para = {extra_disc_para})', 'rb')
    sim_result = pickle.load(sim_file)      # Load data in sim_file to the program
    sim_file.close()
    tot_cost_list.append(sim_result["tot_cost"])
    perform_bound_list.append(sim_result["performance_bound"])

sim_data = pd.DataFrame({
    "epsilon": perform_bound_list,
    "tot_cost": tot_cost_list
})

# Export DataFrame as an xlsx doc
sim_data.to_excel('sim_data.xlsx')

