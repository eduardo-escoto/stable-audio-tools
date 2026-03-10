class DiTClassifier:
    
    def __init__(self):
        pass


"""
Should take in a model checkpoint, and a new model, and load weights
whereever the model dict matches. Can be incomplete or have no matches, but just 
a loose way to bootstrap the DiTClassifier from an original DIT checkpoint
"""
def unstrict_load_checkpoint_weights():
    pass

"""
This should take in a model specification and checkpoint of a full 
Diffusion Transformer model and create a DiT classifier that 
is about half of the model, and puts a classifier head on top,
and then also loads the model weights from the original DiT checkpoint.
"""
def create_half_ditclass_from_full():
    pass