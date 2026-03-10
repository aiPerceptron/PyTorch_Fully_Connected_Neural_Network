# Scoring #5
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
#set y_cat to y_cat.cpu().numpy()
acc_score = accuracy_score(y_concat.cpu().numpy(), y_hat_concat.cpu().numpy())
#print(acc_score)

p_score = precision_score(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy(),average="macro")

#print(p_score)

r_score = recall_score(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy(),average="macro")

#print(r_score)

f_score = f1_score(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy(),average="macro")

#print(f_score)

average_score = (acc_score + p_score + r_score + f_score)/4*100
print(average_score)

c_matrix = confusion_matrix(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy())

disp = ConfusionMatrixDisplay(c_matrix)
disp.plot()
c_matrix
