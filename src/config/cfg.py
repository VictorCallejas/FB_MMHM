# RUN CONFIG
run = dict( 
    setup = 'level-0',
    model = 'Bert', #Bert or MM
    model_name='10_er', # Saved model name
    device = 'cuda',
    n_folds = 10,
    submission = False  # DO JUST WITH FINAL TRAIN, will use last epoch model, this is to make subbmission without ensemble
)

# REPRODUCIBILITY
seed = 42 

# DATA
data_directory = '../data/folds/' + str(run['n_folds']) + '/'
img_path = 'raw/'

final_train = False # IF FINAL TRAIN NO VALIDATION, train=train+dev, dev=0, USE n_folds = 1
