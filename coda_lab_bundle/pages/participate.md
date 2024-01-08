# How to participate

You can submit a solution for only one or several challenges at a time. 
For the 'OOD generalization' and 'Finding Hidden Capabilities challenges' you 
should submit a single script called `submission.py` containing the labeling function 
for the challenge. The labeling functions must be called `predict_labels_binary_ood` and
`predict_labels_keyval_backdoors`, otherwise the scoring program won't recognize them. 
For the 'Repair an Ablated Circuit' challenge you must submit a pt file with the model
repaired model weights called `palindrome_repair01.pt`. 

For more details on the challenges check the [announcement LessWrong post](https://www.lesswrong.com/posts/SbLRsdajhMQdAbaqL/capture-the-flag-mechanistic-interpretability-challenges). For a demo of how
import start working on the challenges check the starting kit (also available as a [GitHub repo](https://github.com/AlejoAcelas/Mech-Interp-Challenges)). 