from functions.eval import Evaluator

def eval(model, data_loader, mean_data, checkpoint=None):

    return Evaluator.run(model, 
                        data_loader,
                        mean_data, 
                        checkpoint, 
                        )
