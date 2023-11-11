# multimodal-sentiment-analysis

## Summary
Στο παρόν project επιχειρείται η ανάλυση συναισθήματος στο [MVSA-Single dataset](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/), το οποίο περιέχει κείμενα και εικόνες. Εποπτικά το προτεινόμενο σύστημα φαίνεται στην εικόνα: ![proposed multimodal system](./images/pipeline.jpg)

### Folders
* images: Περιλαμβάνει το σύνολο των εικόνων που χρησιμοποιούνται στο README.
* libraries: Φάκελος που περιέχει τις βιβλιοθήκες που δημιουργήθηκαν και καλούνται από τα .ipynb αρχεία.

### Files
Εδώ θα εξηγηθούν τα αρχεία που βρίσκονται στο συγκεκριμένο repository. Όσα αρχεία έχουν τη κατάληξη .ipynb έτρεχαν κατά τα πειράματα μέσω του Google Colab, με απαραίτητη τη χρήση κάποιας GPU. Συνεργάζονται άμεσα με το Google Drive από το οποίο ζητούν και στέλνουν αρχεία. Όσα αρχεία έχουν τη κατάληξη .py έτρεχαν locally στον υπολογιστή, χωρίς σημαντικές απαιτήσεις υπολογιστικών πόρων. Σε αυτή τη κατηγορία έχουμε τα αρχεία συναρτήσεων που χρησιμοποιούμε σαν βιβλιοθήκες και τα αρχεία που χρησιμοποιούμε για την εξαγωγή διαγραμμάτων και για τη μετάφραση των κειμένων.
* Konstantinos_gerogiannis_thesis.pdf: Αποτελεί το κείμενο της διπλωματικής εργασίας, όπου περιγράφονται αναλυτικά η θεωρία και τα πειράματα που πραγματοποιήθηκαν. Το αρχείο αυτό είναι γραμμένο στα ελληνικά.
* Konstantinos_gerogiannis_presentation.pptx: Η παρουσίαση της διπλωματικής εργασίας δίνοντας έμφαση στα κυριότερα σημεία της.

## Step 1: Dataset
Αρχικά θα πρέπει να κατεβάσετε το dataset MVSA-Single από το link που υπάρχει στην αρχή της εισαγωγής. Τo dataset θα βρίσκεται σε ένα αρχείο με τον τίτλο MVSA-Single.zip, το οποίο το ανεβάζουμε στο Google Drive ώστε να το επεξεργαζόμαστε άμεσα από το Google Colab. Το dataset είναι βρώμικο και χρειάζεται να γίνει κάποιος καθαρισμός του προτού είναι έτοιμο για χρήση. Συγκεκριμένα πρέπει να εξαχθεί το συνολικό label των ζευγών κειμένου-εικόνας. Αυτό ακριβώς πραγματοποιείται στο notebook με όνομα MVSA dataset.ipynb, όπου μετά την προεπεξεργασία θα πρέπει το dataset να περιέχει 4511 ζεύγη αντί για 4869 που είχε αρχικά. Επίσης το notebook δίνει τη δυνατότητα στο χρήστη να εξάγει αν θέλει μόνο τη πληροφορία που σχετίζεται με τα κείμενα ή με την εικόνα, αντί για τη συνολική. Το συγκεκριμένο notebook μπορεί να χρησιμοποιηθεί και για τον καθαρισμό του MVSA-Multiple, όπου με σωστή εκτέλεση θα προκύψουν 17024 ζεύγη στο αρχείο εξόδου. Ωστόσο η δουλειά αυτή έχει ήδη γίνει και το καθαρισμένο dataset βρίσκεται στα συνοδευτικά αρχεία που χρειάζεται να ανέβουν στο Google Drive.

## Step 2: Predictions from Vader tool (Optional)
Αυτό το βήμα είναι τελείως προαιρετικό καθώς φάνηκε στη πορεία των πειραμάτων πως το συγκεκριμένο εργαλείο που βασίζεται στην απόδοση συναισθήματος με βάση κάποιο λεξικό δε καταφέρνει να ενισχύσει τις συνολικές προβλέψεις του συστήματος. Παρόλα αυτά, διατίθεται το vader_sentiment_scores.ipynb αρχείο που περιέχει τον κώδικα για να αναπαραχθούν οι προβλέψεις του. Οι υπολογισμοί αυτοί έχουν γίνει για όλα τα κείμενα που μας ενδιαφέρουν και βρίσκονται στο φάκελο με τα αρχεία του Google Drive, με όνομα vader_values.npy.

## Step 3: Understanding created libraries
Οι βιβλιοθήκες που δημιουργήθηκαν βρίσκονται στον φάκελο με τα αρχεία που πρέπει να ανέβουν στο Drive, στον φάκελο με όνομα Functions. Στον φάκελο αυτό υπάρχουν 4 διαφορετικά αρχεία:
* text_functions.py: Συναρτήσεις που σχετίζονται με την προεπεξεργασία των κειμένων
* image_functions.py: Συναρτήσεις που σχετίζονται με την προεπεξεργασία των εικόνων
* multimodal_functions.py: Συναρτήσεις που σχετίζονται με το μοντέλο συνένωσης. Περιέχει τις κλάσεις των 4 προτεινόμενων αρχιτεκτονικών συνένωσης και την αρχικοποίηση του μοντέλου που επιλέγουμε.
* sentiment_analysis_functions.py: Συναρτήσεις που χειρίζονται τη συνολική λειτουργία του συστήματος. Εδώ ορίζονται και αρχικοποιούνται τα μοντέλα κειμένου και εικόνας. Επίσης γίνεται η εκμάθηση των μοντέλων, η πρόβλεψη των μοντέλων, η εξαγωγή των χαρακτηριστικών, η δημιουργία των dataloaders.

## Step 4: Create the multimodal system
Όλη η διαδικασία της εκπαίδευσης και δημιουργίας προβλέψεων του συστήματος πραγματοποιείται στο notebook sentiment_analysis_full_process.ipynb. Στο ίδιο notebook εκτελούνται και όλα τα υπόλοιπα πειράματα που αναφέρονται στη διπλωματική και θα εξηγηθούν ένα προς ένα στη συνέχεια. Όπου υπάρχει η σήμανση optional σημαίνει πως η χρήση του συγκεκριμένου κώδικα είναι προαιρετική και επιλέγεται ανάλογα με το πείραμα που θέλουμε να εκτελέσουμε.
