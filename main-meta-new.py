import os
import argparse
import config
import metatrain_3 as metatrain
import eval
import random
import train
import torch
torch.manual_seed(1)
def _train(training_projects,validating_project,testing_project,lr,vocab_file_path=None, model_file_path=None):
    print('\nStarting the training process......\n')

    if vocab_file_path:
        code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
        print('Vocabulary will be built by given file path.')
        print('\tsource code vocabulary path:\t', os.path.join(config.vocab_dir, code_vocab_path))
        print('\tast of code vocabulary path:\t', os.path.join(config.vocab_dir, ast_vocab_path))
        print('\tcode comment vocabulary path:\t', os.path.join(config.vocab_dir, nl_vocab_path))
    else:
        print('Vocabulary will be built according to dataset.')

    if model_file_path:
        print('Model will be built by given state dict file path:', os.path.join(config.model_dir, model_file_path))
    else:
        print('Model will be created by program.')

    print('\nInitializing the training environments......\n')
    train_instance = metatrain.MetaTrain(training_projects=training_projects,validating_project=validating_project,vocab_file_path=vocab_file_path, model_file_path=model_file_path,lr=lr,save_path=f'{testing_project}_meta_3')
    print('Environments built successfully.\n')
    print('Size of train dataset:', train_instance.meta_datasets_size)

    config.logger.info('Size of train dataset:{}'.format(train_instance.meta_datasets_size))
    config.logger.info('Size of source code vocabulary: {}'.format(train_instance.code_vocab_size))
    config.logger.info('Size of ast of code vocabulary: {}'.format(train_instance.ast_vocab_size))
    config.logger.info('Size of code comment vocabulary: {} '.format(train_instance.nl_vocab_size))

    if config.validate_during_train:
        print('\nValidate every', config.validate_every, 'batches and each epoch.')
        #print('Size of validation dataset:', train_instance.eval_instance.dataset_size)
        #config.logger.info('Size of validation dataset: {}'.format(train_instance.eval_instance.dataset_size))

    print('\nStart training......\n')
    config.logger.info('Start training.')
    best_model = train_instance.run_train()
    print('\nTraining is done.')
    config.logger.info('Training is done.')

    # writer = SummaryWriter('runs/CodePtr')
    # for _, batch in enumerate(train_instance.train_dataloader):
    #     batch_size = len(batch[0][0])
    #     writer.add_graph(train_instance.model, (batch, batch_size, train_instance.nl_vocab))
    #     break
    # writer.close()
    del train_instance
    torch.cuda.empty_cache()
    return best_model


def _test(model,vocab_file_path,testing_project,num_of_data=-1):
    dataset_dir = "../dataset_v2/"
    if isinstance(model, dict):
        train_instance = train.Train(vocab_file_path=vocab_file_path, model_state_dict=model,
                                    code_path=os.path.join(dataset_dir,f'original/{testing_project}/train.code')
                                    ,ast_path=os.path.join(dataset_dir,f'original/{testing_project}/train.sbt'),
                                    nl_path=os.path.join(dataset_dir,f'original/{testing_project}/train.comment'),batch_size=config.support_batch_size,
                                    code_valid_path=f'../dataset_v2/original/{testing_project}/valid_transfer.code',nl_valid_path=f'../dataset_v2/original/{testing_project}/valid_transfer.comment',
                                        ast_valid_path=f'../dataset_v2/original/{testing_project}/valid_transfer.sbt'
                                        ,num_of_data=num_of_data)
    elif isinstance(model,str):
        train_instance = train.Train(vocab_file_path=vocab_file_path, model_file_path=model,
                                    code_path=os.path.join(dataset_dir,f'original/{testing_project}/train.code')
                                    ,ast_path=os.path.join(dataset_dir,f'original/{testing_project}/train.sbt'),
                                    nl_path=os.path.join(dataset_dir,f'original/{testing_project}/train.comment'),batch_size=config.support_batch_size,
                                    code_valid_path=f'../dataset_v2/original/{testing_project}/valid_transfer.code',nl_valid_path=f'../dataset_v2/original/{testing_project}/valid_transfer.comment',
                                        ast_valid_path=f'../dataset_v2/original/{testing_project}/valid_transfer.sbt'
                                        ,num_of_data=num_of_data)       
    best_model_test_dict=train_instance.run_train()
    print('\nInitializing the test environments......')
    test_instance = eval.Test(best_model_test_dict,
                              code_path=os.path.join(dataset_dir,f'original/{testing_project}/valid.code')
                                ,ast_path=os.path.join(dataset_dir,f'original/{testing_project}/valid.sbt'),
                                nl_path=os.path.join(dataset_dir,f'original/{testing_project}/valid.comment'))
    print('Environments built successfully.\n')
    config.logger.info('Size of test dataset: {}'.format(test_instance.dataset_size))

    config.logger.info('Start Testing.')
    print('\nStart Testing......')
    test_instance.run_test()
    print('Testing is done.')


def split_dataset(projects):
    random.seed(1)

    testing_project=random.choice(projects)
    projects.remove(testing_project)
    validating_project=random.choice(projects)
    projects.remove(validating_project)
    training_projects = projects

    return (training_projects, validating_project, testing_project)

def list_of_strings(arg):
    return arg.split(',')

if __name__ == '__main__':
    # projects = ['AppScale/appscale','edx/edx-platform','sympy/sympy','IronLanguages/main','mne-tools/mne-python','JiYou/openstack','openhatch/oh-mainline','cloudera/hue','ahmetcemturan/SFACT','mne-tools/mne-python'] # tạm fix cứng
    # training_projects=['AppScale/appscale','edx/edx-platform','sympy/sympy' ,'JiYou/openstack','IronLanguages/main','openhatch/oh-mainline','mne-tools/mne-python','cloudera/hue']
    # validating_project="ahmetcemturan/SFACT"
    # testing_project="kbengine/kbengine"
    project2sources = {
        'spring-boot': ['spring-framework', 'dubbo', 'flink', 'kafka', 'spring-security', 'guava', 'ExoPlayer'], 
        'spring-framework': ['spring-boot', 'dubbo', 'flink', 'spring-security', 'kafka', 'ExoPlayer', 'guava'], 
        'spring-security': ['spring-framework', 'spring-boot', 'dubbo', 'kafka', 'flink', 'ExoPlayer' ,'guava'], 
        'guava': ['flink', 'dubbo', 'spring-framework', 'kafka', 'ExoPlayer', 'spring-boot', 'dagger'], 
        'ExoPlayer': ['flink', 'spring-framework', 'guava', 'kafka', 'spring-boot', 'dubbo', 'spring-security'], 
        'kafka': ['flink', 'spring-boot', 'spring-framework', 'dubbo', 'guava', 'ExoPlayer', 'spring-security'], 
        'dubbo': ['spring-framework', 'spring-boot', 'flink', 'kafka', 'guava', 'spring-security', 'dagger'], 
        'flink': ['kafka', 'spring-framework', 'dubbo', 'spring-boot', 'guava', 'ExoPlayer', 'spring-security'], 
    }
    project2validate={        
        'spring-boot': 'dagger', 
        'spring-framework': 'dagger', 
        'spring-security': 'dagger', 
        'guava': 'spring-security', 
        'ExoPlayer': 'dagger', 
        'dagger': 'dubbo', 
        'kafka': 'dagger', 
        'dubbo': 'ExoPlayer', 
        'flink': 'dagger', }
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-v', '--validate', type=str,default=None)

    parser.add_argument('-t','--test',
                        type=list_of_strings)
    parser.add_argument('-lr','--learning',type=float,default=0.05)
    args = parser.parse_args()
    # validating_project='dubbo'
    # testing_project='dagger'
    #training_projects=['dubbo','guava','kafka']
    testing_project=args.test
        #training_projects, validating_project, testing_project = split_dataset(projects)
    for project in testing_project:
        config.logger.info(f'validate: {project2validate[project]}, testing: {project}')
        best_model_dict = _train(training_projects=project2sources[project][0:3], \
                                validating_project=project2sources[project][3],\
                                testing_project=project,\
                                vocab_file_path=(config.code_vocab_path, config.ast_vocab_path, config.nl_vocab_path)
                                ,model_file_path='../pretrain_model/pretrain.pt',lr=args.learning)
    #_test(best_model_dict,vocab_file_path=(config.code_vocab_path, config.ast_vocab_path, config.nl_vocab_path),testing_project=testing_project,num_of_data=100)
    
    #  _test(os.path.join('20240511_132257', 'model_valid-loss-3.3848_epoch-14_batch--1.pt'))

