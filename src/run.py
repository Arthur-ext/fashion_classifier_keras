import argparse as ap
from main import run_model, save_model

if __name__ == '__main__':
        
    parser = ap.ArgumentParser()

    parser.add_argument('-s', '--save',
                        help='Save Model as file, need to pass the name of file you want',
                        dest='save',
                        action='store')
    
    parser.add_argument('-e', '--epochs',
                        help='Set value of epochs to the Model',
                        dest='epochs',
                        action='store',
                        type=int)

    args = parser.parse_args()
    
    hist, model = run_model(args.epochs) if args.epochs else run_model()
    
    if args.save:
        save_model(model, args.save)
        