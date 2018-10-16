import click
import xlsxwriter
import os

from model.utils.text_utils import diff_rows

from evaluate_model import insert_image_worksheet, predict_with_image, read_img

model_name = ""
batch_size = 20

def eval_metric(expected_result, result):
    match_n, explanation = diff_rows(expected_result, result)
    match_n_1, explanation_1 = diff_rows(result, expected_result)

    try:
        return (match_n/len(expected_result) + match_n_1/len(result))/2, explanation
    except:
        return 0.0, explanation


"""
# CTC model for testing can be downloaded from: https://drive.google.com/open?id=1eChfYOWNfsuHEwZnT4230iCLd8ZJKOna
# Attention model: https://drive.google.com/open?id=1rF_8DcjY69GmMvM_io9aazc5ZbIXiAtK
"""
@click.command()
@click.option('--results', default="./results_from_server/results1/full/", help='Dir to results')
@click.option("--vocab", default="./results_from_server/results1/full/vocab.txt", help="vocab.txt")
@click.option("--prefix", default="./data1/img-val/", help="prefix paths which used in souce file")
@click.option('--src_test', default="./data1/src-val.txt", help = "Dir to source")
@click.option("--tgt_test", default="./data1/tgt-val.txt", help = "Dir to target")
@click.option("--is_ctc", default=True, help = "Dir to target")
def main(results, vocab, prefix, src_test, tgt_test, is_ctc):
    assert os.path.isfile(src_test) and os.path.isfile(tgt_test)

    img_paths= [os.path.join(prefix, src.strip()) for src in open(src_test).readlines()]
    imgs     = [read_img(img_path)[-1] for img_path in img_paths]
    img_lbls = [_.strip() for _ in open(tgt_test).readlines()]

    predicts = predict_with_image(model_saved_path=results, vocab_path=vocab, imgs=imgs, is_ctc=is_ctc)

    workbook  = xlsxwriter.Workbook('%s_result_from_%s.xlsx' % ('ctc' if is_ctc else 'attn',
                                                                tgt_test.split('/')[-1].replace('.','_')) )
    worksheet = workbook.add_worksheet("Accuracy Report")

    worksheet.write(0, 0, 'Index')
    worksheet.write(0, 1, 'File Name')
    worksheet.write(0, 2, 'Input')
    worksheet.write(0, 3, 'Output')
    worksheet.write(0, 4, 'CorrectAnswer')
    worksheet.write(0, 5, 'Accuracy')
    worksheet.write(0, 6, 'Explanation')

    scale = 5
    index = 0
    for _id, (lbl, predict) in enumerate(zip(img_lbls, predicts)):
        index += 1
        _row_idx = scale*index-scale+1
        acc, explanation = eval_metric(expected_result=lbl, result=predict)

        worksheet.write(_row_idx, 0, index)
        worksheet.write(_row_idx, 1, img_paths[_id])

        #insert_image_worksheet(worksheet, img_paths[_id], index, 2, scale)
        worksheet.write(_row_idx, 3, predict)
        worksheet.write(_row_idx, 4, lbl)
        worksheet.write(_row_idx, 5, acc)
        worksheet.write(_row_idx, 6, explanation)

    workbook.close()

if __name__ == '__main__':
    main()