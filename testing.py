# Testing #4
y_hat_concat = torch.tensor([]).to(torch.float32).to(device)
y_concat = torch.tensor([]).to(torch.float32).to(device)

running_correct_sum = 0
running_sum = 0

model.eval()
with torch.no_grad():
  for xb,yb in dl_test:
    xb,yb = xb.to(device),yb.to(device)
    ml_out_batch = model(xb)
    yb_hat = ml_out_batch.argmax(axis=1) # this line decides what type of class the data is most likely to be
    running_correct_sum += sum(yb_hat == yb)
    running_sum += len(yb)

    y_hat_concat = torch.cat([y_hat_concat, yb_hat], dim=0) # Adding yb_hat to y_hat for heavy data analysis
    y_concat = torch.cat([y_concat, yb], dim=0)

percent_accuracy = (running_correct_sum/running_sum)*100
print(percent_accuracy.item(),"%")

#print(len(y_hat_concat))
#print(len(y_concat))
