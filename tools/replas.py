import numpy as np
import scipy.io as sio
import glob
data_path = '/data/xxinwei/dataset/objectnn_hardest/'
# classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
#                          'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
#                          'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
#                          'person','piano','plant','radio','range_hood','sink','sofa','stairs',
#                          'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
classnames = ['bag', 'bed', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'pillow', 'shelf', 'sink', 'sofa', 'table', 'toilet']
for i in range(len(classnames)):
    #view = sio.loadmat(data_path+classnames[i]+'/train/view.mat')
    #view1 = view['ll']
    #np.save(data_path + classnames[i] + '/train/view.npy', view1)
    #rand_view_num = np.load('/data/xxinwei/view-transformer-hardest/ModelNet40_hardest/'+classnames[i]+'/train/random_view_num.npy')
    numb = glob.glob(data_path + classnames[i]+'/test/*')
    rand_view_num = np.random.randint(6,20,(len(numb)-2)//20)
    np.save(data_path + classnames[i]+'/test/random_view_num.npy',rand_view_num)
    ss  = sio.loadmat(data_path + classnames[i]+'/test/view.mat')
    ss = ss['ll']
    np.save(data_path + classnames[i]+'/test/view.npy',ss)