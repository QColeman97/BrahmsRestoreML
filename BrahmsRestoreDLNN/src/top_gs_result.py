# import numpy as np
import math
import json

grid_search_results_path = '../output_grid_search/'
top_file_vl, min_val_loss = None, None
top_file_l, min_loss = None, None
# F35 - 4096 combos
#   First round: 1 - 130ish
#   Second round: 130 - 264
# F35 - 2048 combos
#   First round: 1 - 132ish
for i in range(1, 200):
    # file_name = grid_search_results_path + 'result_' + str(i) + '_of_4096_noPC.txt'
    file_name = grid_search_results_path + 'result_' + str(i) + '_of_2048_noPC.txt'
    try:
        result_file = open(file_name, 'r')
    except:
        print('GS Result File', i, 'probably doesn\'t exist (open error)')
        break

    curr_vl_str = result_file.readline()
    _ = result_file.readline()
    curr_l_str = result_file.readline()
    _ = result_file.readline()
    curr_val_loss = float(curr_vl_str)
    curr_loss = float(curr_l_str)
    # if i == 253:
    # #     print(file_name)
    # #     print('Curr Loss (9557266): ' + curr_l_str)
    #     print('Type of val loss:', type(curr_vl_str), type(curr_val_loss), curr_val_loss)

    # curr_val_loss = float(result_file.readline())
    # _ = result_file.readline()
    # curr_loss = float(result_file.readline())
    # _ = result_file.readline()
    # for _ in range(3):
    #     _ = result_file.readline()
    hp_config = json.loads(result_file.readline())
    if hp_config['gamma'] == 0.05:
        if (min_val_loss is None or 
                (curr_val_loss < min_val_loss) or
                (math.isnan(min_val_loss) and not math.isnan(curr_val_loss))):
                # (min_val_loss == np.nan and curr_val_loss != np.nan)):

            min_val_loss = curr_val_loss
            top_file_vl = i
        if (min_loss is None or 
                (curr_loss < min_loss) or
                # (min_loss == np.nan and curr_loss != np.nan)):
                (math.isnan(min_loss) and not math.isnan(curr_loss))):
            min_loss = curr_loss
            top_file_l = i

print('Grid Search Result File', top_file_vl, 'is best with val. loss of', min_val_loss)
print('Grid Search Result File', top_file_l, 'is best with loss of', min_loss)