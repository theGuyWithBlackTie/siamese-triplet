This repository is about contrastive learning. It features contrastive loss, triplet loss with the obvious siamese networks.

Dataset used is MNIST digit images.

### Contrastive Loss
In contrastive loss, siamese networks takes two inputs. These inputs can either belong to same class (y=1) else difference class (y=0). Siamese Networks or rather contrastive loss would try to bring all the input embeddings of one class together and distant far away from other classes.

### Triplet Loss
In triplet loss, siamese networks takes 3 inputs. One input is an anchor input, second input belongs to same class as anchor input and third input belongs to different class from anchor. Triplet loss would also try to bring all the input embedding of one class together closer and distant far away from other classes.

Intuitively, triplet loss is better than contrastive loss. Triplet loss, while defining the region for each class in embedding space, it is aware of atleast one more class. It is same as saying model is aware of all of the dataset while defining the region for each class embeddings.

