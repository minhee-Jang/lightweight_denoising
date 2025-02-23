import os

if __name__ == '__main__':
    seq_list = os.listdir('sequences/')
    # print('seq_list:', seq_list)

    all_vid_list = []
    train_list_f = 'sep_trainlist.txt'
    test_list_f = 'sep_testlist.txt'
    with open(train_list_f, 'r') as f:
        video_list = f.readlines()
        train_list = [l[:-1] for l in video_list]

    with open(test_list_f, 'r') as f:
        video_list = f.readlines()
        test_list = [l[:-1] for l in video_list]

    for s in seq_list:
        sd = os.path.join('sequences', s)
        vid_list = os.listdir(sd)
        # print('vid_list:', vid_list)
        for v in vid_list:
            vd = os.path.join(sd, v)
            if os.path.isdir(vd):
                # print(f'{s}/{v}')
                all_vid_list.append(f'{s}/{v}')


    val_list = set(all_vid_list) - set(train_list)
    val_list = val_list - set(test_list)
    val_list = sorted(list(val_list))

    with open('sep_vallist.txt', 'w') as f:
        for v in val_list:
            print(v)
            f.write(f'{v}\n')