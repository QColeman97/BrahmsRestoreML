import json

grid_search_results_path = '../output_grid_search/'
top_file, min_loss = None, None
for i in range(1, 201):
    file_name = grid_search_results_path + 'result_' + str(i) + '_of_4096_noPC.txt'
    try:
        result_file = open(file_name, 'r')
    except:
        print('GS Result File', i, 'probably doesn\'t exist (open error)')
        break
    curr_val_loss = float(result_file.readline())
    for _ in range(3):
        _ = result_file.readline()
    hp_config = json.loads(result_file.readline())
    if hp_config['gamma'] == 0.05:
        if min_loss is None or curr_val_loss < min_loss:
            min_loss = curr_val_loss
            top_file = i

print('Grid Search Result File', top_file, 'is best with val. loss of', min_loss)