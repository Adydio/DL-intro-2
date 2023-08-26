import torch, random

def get_timestep_data_iter(*data, batch_size, time_step, device, **kwargs):
    has_label = len(data) == 6
    if has_label:
        T, I, X, X_selected, msk, label = data
    else:
        T, I, X, X_selected, msk = data

    offset = random.randint(0, time_step - 1)
    total_timesteps = T.shape[0]
    total_stocks = T.shape[1]

    for t in range(offset, total_timesteps - time_step - 2, time_step):
        for start_stock in range(0, total_stocks, batch_size):
            end_stock = start_stock + batch_size

            batch_T = T[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_I = I[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_X = X[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_X_selected = X_selected[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_msk = msk[t:t+time_step, start_stock:min(end_stock, total_stocks)]

            batch_T = batch_T.permute(1, 0) if len(batch_T.shape) == 2 else batch_T.permute(1, 0, 2)
            batch_I = batch_I.permute(1, 0) if len(batch_I.shape) == 2 else batch_I.permute(1, 0, 2)
            batch_X = batch_X.permute(1, 0) if len(batch_X.shape) == 2 else batch_X.permute(1, 0, 2)
            batch_X_selected = batch_X_selected.permute(1, 0) if len(batch_X_selected.shape) == 2 else batch_X_selected.permute(1, 0, 2)
            batch_msk = batch_msk.permute(1, 0) if len(batch_msk.shape) == 2 else batch_msk.permute(1, 0, 2)

            if has_label:
                batch_label = label[t:t+time_step, start_stock:min(end_stock, total_stocks)]
                batch_label = batch_label.permute(1, 0) if len(batch_label.shape) == 2 else batch_label.permute(1, 0, 2)

                y = batch_label
                y_msk = batch_msk
                X_day_1 = X_selected[t+1:t+time_step+1, start_stock:min(end_stock, total_stocks)]
                msk_day_1 = msk[t+1:t+time_step+1, start_stock:min(end_stock, total_stocks)]
                X_day_2 = X_selected[t+2:t+time_step+2, start_stock:min(end_stock, total_stocks)]
                msk_day_2 = msk[t+2:t+time_step+2, start_stock:min(end_stock, total_stocks)]

                X_day_1 = X_day_1.permute(1, 0) if len(X_day_1.shape) == 2 else X_day_1.permute(1, 0, 2)
                msk_day_1 = msk_day_1.permute(1, 0) if len(msk_day_1.shape) == 2 else msk_day_1.permute(1, 0, 2)
                X_day_2 = X_day_2.permute(1, 0) if len(X_day_2.shape) == 2 else X_day_2.permute(1, 0, 2)
                msk_day_2 = msk_day_2.permute(1, 0) if len(msk_day_2.shape) == 2 else msk_day_2.permute(1, 0, 2)

                y_tuple = (y, y_msk, X_day_1, msk_day_1, X_day_2, msk_day_2)
                yield batch_T, batch_I, batch_X, batch_X_selected, batch_msk, y_tuple
            else:
                yield batch_T, batch_I, batch_X, batch_X_selected, batch_msk
