from volleyball import *

import pickle


def return_dataset(cfg):
    if cfg.dataset_name=='volleyball':
        train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs)
        train_frames = volley_all_frames(train_anns)
        # pickle.dump(train_anns,open('/home/hh9665/Desktop/CurrentProject/Group-Activity-Recognition/data/volleyball/all_anns_my_dataset','wb'))
        # pickle.dump(train_frames, open(
        #     '/home/hh9665/Desktop/CurrentProject/Group-Activity-Recognition/data/volleyball/train_frames_my_dataset', 'wb'))
        test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames = volley_all_frames(test_anns)
        # pickle.dump(test_anns, open('/home/hh9665/Desktop/CurrentProject/Group-Activity-Recognition/data/volleyball/test_anns_my_dataset', 'wb'))
        # pickle.dump(test_frames, open('/home/hh9665/Desktop/CurrentProject/Group-Activity-Recognition/data/volleyball/test_frames_my_dataset','wb'))
        all_anns = {**train_anns, **test_anns}
        #all_tracks = pickle.load(open('data/volleyball/tracks_normalized.pkl', 'rb'))
        # all_tracks = pickle.load(open('data/volleyball/tracks_normalized_my_dataset', 'rb'))
        all_winner_tracks = pickle.load(open('data/volleyball/tracks_normalized_my_dataset_Recognition', 'rb'))
        all_tracks = pickle.load(open('tracks_normalized_my_dataset_clip.pkl', 'rb'))

        train_winner_frames = volley_winner_extract(train_frames, all_anns)
        test_winner_frames = volley_winner_extract(test_frames, all_anns)

        training_set=VolleyballDataset(all_anns,all_winner_tracks,all_tracks,train_winner_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,num_before=cfg.num_before,
                                       num_after=cfg.num_after,is_training=True,is_finetune=(cfg.training_stage==1))

        validation_set=VolleyballDataset(all_anns,all_winner_tracks,all_tracks,test_winner_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1))

        validation_set1=VolleyballDataset(all_anns,all_winner_tracks,all_tracks,test_winner_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,OR = 0.1, num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1))

        validation_set2=VolleyballDataset(all_anns,all_winner_tracks,all_tracks,test_winner_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,OR = 0.4, num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1))

        validation_set3=VolleyballDataset(all_anns,all_winner_tracks,all_tracks,test_winner_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,OR = 0.7, num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1))
                              
    else:
        assert False
                                         
    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    
    return training_set, validation_set1, validation_set2, validation_set3, validation_set
    
