import argparse
from XAI_PreProcessing import DataProcessing
from XAI_IterativeWeights import IterativeWeights
from XAI_ELM import XAI
from XAI_Plotter import Plotter


def main(benign_path=None, malign_path=None):
   
    # Load benign and malign data

    preProcesser = DataProcessing(benign_path, malign_path)
    if benign_path != None and malign_path != None:
        dataset = preProcesser.create_dataset()
    else:
        dataset, T, P, TVP = preProcesser.get_sample_datasets('linear')

    print('Dataset loaded')
    breakpoint()
    # Calculate Weights

    weight_factory = IterativeWeights(dataset, 100)
    #weights_xai = weight_factory.get_xai_weights()
    weights_elm = weight_factory.get_random_weights(NumberofHiddenNeurons=100)


    # Run XAI algorithm

    xai = XAI(dataset)
    #xai.run_xai_elm()

    # Run traditional ELM algorithm
    xai.run_traditional_elm(
        weights= weights_elm, 
        NumberofHiddenNeurons= 50,
        ActivationFunction= 'dilation')

    # Plot the data
    plotter = Plotter()
    plotter.plot_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process paths to benign and malign files.')
    parser.add_argument('--benign', type=str, help='Path to benign file', default=None, required=False)
    parser.add_argument('--malign', type=str, help='Path to malign file', default=None, required=False)
    
    args = parser.parse_args()
    
    main(benign_path=args.benign, malign_path=args.malign)