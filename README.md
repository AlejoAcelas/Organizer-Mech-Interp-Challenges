# CTF Mechanistic Interpretability Challenges (Organizer)  

This repository contains the materials used to create the Capture The Flag Mechanistic Interpretability Challenges (see [here](https://www.lesswrong.com/posts/SbLRsdajhMQdAbaqL/capture-the-flag-mechanistic-interpretability-challenges) for a description). **Do not check it out unless you're OK with being spoiled the answer to the challenges**. If you want to take a stab at the challenges, check out the [participant's repository](https://github.com/AlejoAcelas/Mech-Interp-Challenges) which contains the starting materials for the competition.  

Some files and directories you might want to check are:  

* `dataset.py`: Contains the data generation and labeling functions used to train the challenge models. In particular, the labeling function for the Binary Addition and Key-Value pairs challenges correspond to the expected answer in each of those challenges.  

* `coda_lab_bundle/`: Bundle containing formatted for the CodaLab platform
  * `scoring/scoring.py`: Script used to grade challenge submissions
  * `scoring/palindrome_classifier.pt`: Unablated model from the Palindrome Challenge. It corresponds to one of the possible submissions that would have obtained a perfect score for that challenge.
