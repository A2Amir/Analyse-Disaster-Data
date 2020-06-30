
    
from utils import *
import argparse
    
def main(f1_dir):
    
    df, X, y, category_names = load_data(f1_dir)
    #X = X[:200]
    #y = y[:200]
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print('The training set shape {} and the grount truth shape {}'.format(X_train.shape, y_train.shape))
    print('The testing set shape {} and the grount truth shape {}'.format(X_test.shape, y_test.shape))

    model = build_model(X_train,y_train)


    # To show the accuracy, precision, and recall of the tuned model.  
    y_pred = model.predict(X_test)
    overall_accuracy = (y_pred == y_test).mean().mean()
    fb_score = get_fbeta_score(y_test, y_pred)

    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('Fbeta score {0:.2f}%\n'.format(fb_score*100))

    dump_model(model)
    




if __name__ == "__main__":
    
    
    path = os.getcwd()


    parser = argparse.ArgumentParser(description='train classifier')

    parser.add_argument('--f1', help='The file name and address of the database file (example : ../data/DisasterResponse.db).')


    args = parser.parse_args()

    if not args.f1:
        raise ImportError('The --f1 parameter needs to be provided (example: ../data/DisasterResponse.db)')
    else:
        f1_dir = args.f1


    main(f1_dir)



