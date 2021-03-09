# import numpy as np
import math
import json


def min_file_and_value(results_path, num_iters, val_loss=False, pc=False):
    min_file, min_value = None, None

    print('OUTPUT GS PATH:', results_path)

    for i in range(1, num_iters + 1):
        pc_suffix = '.txt' if pc else '_noPC.txt'
        file_name = results_path + 'result_' + str(i) + '_of_' + str(num_iters) + pc_suffix
        result_file = open(file_name, 'r')

        curr_vl_str = result_file.readline()
        _ = result_file.readline()
        curr_l_str = result_file.readline()
        _ = result_file.readline()

        if len(curr_vl_str) == 0:
            continue
        # print(i, len(curr_vl_str))
        # print(i, len(curr_l_str))

        curr_val_loss = float(curr_vl_str)
        curr_loss = float(curr_l_str)

        # hp_config = json.loads(result_file.readline())
        if val_loss:
            if (min_value is None or 
                    (curr_val_loss < min_value) or
                    (math.isnan(min_value) and not math.isnan(curr_val_loss))):

                min_value = curr_val_loss
                min_file = i

        else:
            if (min_value is None or 
                    (curr_loss < min_value) or
                    (math.isnan(min_value) and not math.isnan(curr_loss))):

                min_value = curr_loss
                min_file = i

    return min_file, min_value


# Returns result files w/in 10% of min loss - work until yields ~20 files
def files_close_to_min(within, min_val, results_path, num_iters, val_loss=False, pc=False):
    within_min_files, within_min_values = [], []
    # within = 0.25 # 0.1
    # value_limit_vl = min_val + (min_val * within_vl)
    value_limit = min_val + (abs(min_val) * within)
    print('Min val:', min_val, 'val limit:', value_limit)

    # print('Value limit:', value_limit)

    for i in range(1, num_iters + 1):
        pc_suffix = '.txt' if pc else '_noPC.txt'
        file_name = results_path + 'result_' + str(i) + '_of_' + str(num_iters) + pc_suffix
        result_file = open(file_name, 'r')

        curr_vl_str = result_file.readline()
        _ = result_file.readline()
        curr_l_str = result_file.readline()
        _ = result_file.readline()

        if len(curr_vl_str) == 0:
            continue

        # # TEMP - filter among gamma
        # hp_config = json.loads(result_file.readline())
        # if hp_config['gamma'] != 0.05:
        #     continue

        if val_loss:
            curr_val_loss = float(curr_vl_str)
            # if curr_val_loss <= value_limit_vl:
            if curr_val_loss <= value_limit:
                within_min_files.append(i)
                within_min_values.append(curr_val_loss)
        else:
            curr_loss = float(curr_l_str)
            if curr_loss <= value_limit:
                within_min_files.append(i)
                within_min_values.append(curr_loss)

    return within_min_files, within_min_values


# def top_file_iterate(top_files, num_iters, val_loss=False, single=False):
#     grid_search_results_path = '../output_grid_search/'
#     # grid_search_results_path = '../output_grid_search_save_11_19/'
#     top_file, min_loss = None, None
    
#     for i in range(1, num_iters + 1):
#         file_name = grid_search_results_path + 'result_' + str(i) + '_of_4096_noPC.txt'
#         # file_name = grid_search_results_path + 'result_' + str(i) + '_of_2048_noPC.txt'
#         try:
#             result_file = open(file_name, 'r')
#         except:
#             print('GS Result File', i, 'probably doesn\'t exist (open error)')
#             break

#         curr_vl_str = result_file.readline()
#         _ = result_file.readline()
#         curr_l_str = result_file.readline()
#         _ = result_file.readline()
#         curr_val_loss = float(curr_vl_str)
#         curr_loss = float(curr_l_str)

#         # # Add to top val loss
#         # if (i not in top10perc_vl_files) and (curr_top_index_vl < max_top_range):

#         hp_config = json.loads(result_file.readline())
#         if hp_config['gamma'] == 0.05:

#             if val_loss:
#                 if ((min_loss is None or 
#                         (curr_val_loss < min_loss) or
#                         (math.isnan(min_loss) and not math.isnan(curr_val_loss))) and
#                         i not in top_files):

#                     min_loss = curr_val_loss
#                     top_file = i

#             else:
#                 if ((min_loss is None or 
#                         (curr_loss < min_loss) or
#                         (math.isnan(min_loss) and not math.isnan(curr_loss))) and
#                         i not in top_files):

#                     min_loss = curr_loss
#                     top_file = i

#     return (top_file, min_loss)



def main():
    # OLD
    # F35 - 4096 combos
    #   First round: 1 - 130ish
    #   Second round: 130 - 264
    #   Third round: 265 - 
    # F35 - 2048 combos
    #   First round: 1 - 132ish

    # CURRENT
    # F35 wb (rnn) - 3072 combos
    # F35 (lstm) - 192 combos
    # PC wb (rnn) - 2048 combos
    # PC bvs (rnn) - 24 combos
    # PC low tsteps 2 - 72 combos
    num_gs_iters = 72
    pc = True

    if num_gs_iters == 3072:
        grid_search_results_path = 'brahms_restore_ml/drnn/output_grid_search_wb/'
        within_vl, within_l = 6.5, 0.25
    elif num_gs_iters == 2048:
        grid_search_results_path = 'brahms_restore_ml/drnn/output_grid_search_pc_wb/'
        within_vl, within_l = 0.15, 0.05   
    elif num_gs_iters == 192:
        grid_search_results_path = 'brahms_restore_ml/drnn/output_grid_search_lstm/'
        within_vl, within_l = 0.7, 0.6
    elif num_gs_iters == 24:
        grid_search_results_path = 'brahms_restore_ml/drnn/output_grid_search_bvs/'
        within_vl, within_l = 0.7, 0.6
    elif num_gs_iters == 72:
        grid_search_results_path = 'brahms_restore_ml/drnn/output_grid_search_low_tsteps2/'
        within_vl, within_l = 10.0, 0.6   # gamma = 0.3
        # within_vl, within_l = 100000.0, 100000.0    # gamma = 0.05, 0.15
    else:
        grid_search_results_path = 'brahms_restore_ml/drnn/output_grid_search/'
        within_vl, within_l = 0.1, 0.1

    # # top10perc_vl_files, top10perc_l_files = [], []
    # # top10perc_vl, top10perc_l = [], []
    # min_vl_files, min_l_files = [], []
    # min_vl_values, min_l_values = [], []
    # # curr_top_index_vl, curr_top_index_l, max_top_range = 0, 0, int(num_gs_iters * 0.1)

    # min_val_loss, min_loss, min_vl_file, min_l_file = None, None, 0, 0

    min_vl_file, min_val_loss = min_file_and_value(grid_search_results_path, num_gs_iters, val_loss=True, pc=pc)
    min_l_file, min_loss = min_file_and_value(grid_search_results_path, num_gs_iters, pc=pc)

    print('Min val. loss:', min_val_loss, min_vl_file)
    print('Min loss:', min_loss, min_l_file)

    min_vl_files, min_vl_values = files_close_to_min(within_vl, min_val_loss, grid_search_results_path, num_gs_iters, val_loss=True, pc=pc)
    min_l_files, min_l_values = files_close_to_min(within_l, min_loss, grid_search_results_path, num_gs_iters, pc=pc)

    with open('brahms_restore_ml/drnn/top_gs_results_vl_' + str(num_gs_iters) + '.txt', 'w') as fp:
        for i, file_num in enumerate(min_vl_files):
            line_string = str(file_num) + ', ' + str(min_vl_values[i]) + '\n'
            fp.write(line_string)
    with open('brahms_restore_ml/drnn/top_gs_results_l_' + str(num_gs_iters) + '.txt', 'w') as fp:
        for i, file_num in enumerate(min_l_files):
            line_string = str(file_num) + ', ' + str(min_l_values[i]) + '\n'
            fp.write(line_string)

    # OLD BELOW - min 10 percent out of total results

    # while (curr_top_index_vl < max_top_range) or (curr_top_index_l < max_top_range):

    #     if curr_top_index_vl < max_top_range:
    #         new_top_vlfile, new_top_vl = top_file_iterate(top10perc_vl_files, num_gs_iters, val_loss=True)
    #         top10perc_vl_files.append(new_top_vlfile)
    #         top10perc_vl.append(new_top_vl)
    #         curr_top_index_vl += 1

    #     if curr_top_index_l < max_top_range:
    #         new_top_lfile, new_top_l = top_file_iterate(top10perc_l_files, num_gs_iters)
    #         top10perc_l_files.append(new_top_lfile)
    #         top10perc_l.append(new_top_l)
    #         curr_top_index_l += 1

    # # print('Top vl files:\n', len(top10perc_vl_files))
    # # print('Top val losses:\n', top10perc_vl)
    # # print('Top l files:\n', len(top10perc_l_files))
    # # print('Top losses:\n', top10perc_l)

    # with open('../top_gs_results_vl.txt', 'w') as fp:
    #     for i in range(max_top_range):
    #         line_string = str(top10perc_vl_files[i]) + ', ' + str(top10perc_vl[i]) + '\n'
    #         fp.write(line_string)
    # with open('../top_gs_results_l.txt', 'w') as fp:
    #     for i in range(max_top_range):
    #         line_string = str(top10perc_l_files[i]) + ', ' + str(top10perc_l[i]) + '\n'
    #         fp.write(line_string)


    # OLDER BELOW

    #     for i in range(1, num_iters + 1):
    #         file_name = grid_search_results_path + 'result_' + str(i) + '_of_4096_noPC.txt'
    #         # file_name = grid_search_results_path + 'result_' + str(i) + '_of_2048_noPC.txt'
    #         try:
    #             result_file = open(file_name, 'r')
    #         except:
    #             print('GS Result File', i, 'probably doesn\'t exist (open error)')
    #             break

    #         curr_vl_str = result_file.readline()
    #         _ = result_file.readline()
    #         curr_l_str = result_file.readline()
    #         _ = result_file.readline()
    #         curr_val_loss = float(curr_vl_str)
    #         curr_loss = float(curr_l_str)

    #         # Add to top val loss
    #         if (i not in top10perc_vl_files) and (curr_top_index_vl < max_top_range):
    #             # if i == 253:
    #             # #     print(file_name)
    #             # #     print('Curr Loss (9557266): ' + curr_l_str)
    #             #     print('Type of val loss:', type(curr_vl_str), type(curr_val_loss), curr_val_loss)

    #             # curr_val_loss = float(result_file.readline())
    #             # _ = result_file.readline()
    #             # curr_loss = float(result_file.readline())
    #             # _ = result_file.readline()
    #             # for _ in range(3):
    #             #     _ = result_file.readline()

    #             hp_config = json.loads(result_file.readline())
    #             if hp_config['gamma'] == 0.05:
    #                 if (min_val_loss is None or 
    #                         (curr_val_loss < min_val_loss) or
    #                         (math.isnan(min_val_loss) and not math.isnan(curr_val_loss))):
    #                         # (min_val_loss == np.nan and curr_val_loss != np.nan)):

    #                     min_val_loss = curr_val_loss
    #                     top_file_vl = i

    #                     top10perc_vl.append(min_val_loss)
    #                     top10perc_vl_files.append(top_file_vl)
    #                     curr_top_index_vl += 1

    #         # Add to top loss
    #         if (i not in top10perc_vl_files) and (curr_top_index_l < max_top_range):
    #             hp_config = json.loads(result_file.readline())
    #             if hp_config['gamma'] == 0.05:
    #                 if (min_loss is None or 
    #                         (curr_loss < min_loss) or
    #                         # (min_loss == np.nan and curr_loss != np.nan)):
    #                         (math.isnan(min_loss) and not math.isnan(curr_loss))):
    #                     min_loss = curr_loss
    #                     top_file_l = i

    #                     top10perc_l.append(min_loss)
    #                     top10perc_l_files.append(top_file_l)
    #                     curr_top_index_l += 1


    # # top10perc_vl_all = zip(top10perc_vl_files, top10perc_vl)
    # # top10perc_l_all = zip(top10perc_l_files, top10perc_l)

    # # print('Grid Search Result File', top_file_vl, 'is best with val. loss of', min_val_loss)
    # # print('Grid Search Result File', top_file_l, 'is best with loss of', min_loss)

    # # with open('top_gs_results_vl.txt', 'w') as fp:
    # #     for i in range(max_top_range):
    # #         line_string = str(top10perc_vl_files[i]) + ', ' + str(top10perc_vl[i])
    # #         fp.write(line_string)
    # # with open('top_gs_results_l.txt', 'w') as fp:
    # #     for i in range(max_top_range):
    # #         line_string = str(top10perc_l_files[i]) + ', ' + str(top10perc_l[i])
    # #         fp.write(line_string)


if __name__ == '__main__':
    main()