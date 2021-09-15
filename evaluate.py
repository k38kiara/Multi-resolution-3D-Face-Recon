from functions.eval import Evaluator

def eval(model, check_point=None):
    evaluator = Evaluator()
    if check_point:
        model.load(check_point)
        
    return evaluator.run(model)
