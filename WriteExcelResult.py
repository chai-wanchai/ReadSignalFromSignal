import xlsxwriter
import numpy as np
from sklearn.metrics import auc

workbook = xlsxwriter
k = 2

def WriteExcelResult(filename,modelName,epoch,confusion=None):
    global k
    ##########################  Confusion Matrix  #############################

    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.values.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F-meature
    F_measure = 2 * (PPV * TPR) / (PPV + TPR)
    # AUC
    AUC = auc(FPR, TPR, reorder=True)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print(ACC)

    workbook = xlsxwriter.Workbook('C:\\Users\\chai_\\Desktop\\New folder\\'+filename+'.xlsx')
    worksheet = workbook.add_worksheet(name=str(epoch)+'s')
    worksheet.set_column(0,0,width=15.75)

    headerFormat = workbook.add_format({'align':'center','bold': True, 'font_color': 'red','font_size':16,'font_name':'Tahoma'})
    colFormat = workbook.add_format({'bold':True,'align':'center','bg_color':'#FFFF00','font_size':11})
    redBold=workbook.add_format({'bold': True, 'font_color': 'red'})
    center =workbook.add_format({'align':'center'})
    col = []
    for f in range(1,10):
        col.append("K-Fold "+str(f))
    col.append('Trainset')
    col.append('Testset')
    #print(col)
    worksheet.write('A1','Epoch 30 s',headerFormat)
    worksheet.merge_range('B1:C1',modelName,headerFormat)
    for i,data in enumerate(col,start=1):
        worksheet.write(1,i,data,colFormat)
    parameter = ['ACC','TPR','TNR','PPV','NPV','FPR','FNR','FDR', 'F_measure']
    #paraData = [ACC,TPR,TNR,PPV,NPV,FPR,FNR,FDR,F_measure]

    stage=len(ACC)
    AllPara =[]
    i=2
    columnK = k
    for para in parameter:

            worksheet.write(i, 0, para, redBold)
            ii = i
            for s in range(stage):
                ii += 1
                worksheet.write(ii, 0, 'stage' + str(s + 1))
                if para=='ACC':
                    worksheet.write(ii,columnK,ACC[s])
                if para=='TPR':
                    worksheet.write(ii,columnK,TPR[s])
                if para=='TNR':
                    worksheet.write(ii,columnK,TNR[s])
                if para=='PPV':
                    worksheet.write(ii,columnK,PPV[s])
                if para=='NPV':
                    worksheet.write(ii,columnK,NPV[s])
                if para=='FPR':
                    worksheet.write(ii,columnK,FPR[s])
                if para=='FNR':
                    worksheet.write(ii,columnK,FNR[s])
                if para=='FDR':
                    worksheet.write(ii,columnK,FDR[s])
                if para=='F_measure':
                    worksheet.write(ii,columnK,F_measure[s])


            i = i + stage + 2



            worksheet.write(i,0,'AUC',redBold)
            worksheet.write(i+1,0,'Accuracy',redBold)
            worksheet.write(i+2,0,'Record')
            worksheet.write(i+3,0,'Record Train')
            worksheet.write(i+4,0,'Record Test')
    k=+1
    workbook.close()

if __name__ == '__main__':
    workbook = xlsxwriter.Workbook('C:\\Users\\chai_\\Desktop\\New folder\\' + 'lll' + '.xlsx')
    worksheet = workbook.add_worksheet(name=str(555) + 's')
    data = [1,2,3,4,5]
    data2 =[6,5,4]
    h=['kk']
