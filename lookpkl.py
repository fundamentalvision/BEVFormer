import pickle
file='/media/cuhp/SSD/Dataset/nuscenes/v1.0-mini/nuscenes_infos_temporal_train.pkl'
f=open(file,'rb')
content=pickle.load(f)
print(content)
f.close()
