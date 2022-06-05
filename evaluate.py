import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
import faiss
from metrics import select as metricselect

# Evaluate 1 epoch
def eval(net,
        queryloader,
        eval_trainloader,
        kernels,
        session_id,
        reindex,
        prev_gallery_features,
        prev_gallery_labels,
        return_curr_gallery_names = False,
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.eval()
    ################ DEFINE EVALUATION METRICS ##############
    metric_names = [
        'e_recall@1', 'e_recall@2', 'e_recall@4',
        'c_recall@1', 'c_recall@2', 'c_recall@4'
    ]
    list_of_metrics = [metricselect(metricname) for metricname in metric_names]
    requires = [metric.requires for metric in list_of_metrics]
    requires = list(set([x for y in requires for x in y]))
    #################### OBTAIN GALLERY FEATURES ############
    gallery_feature_colls = []
    gallery_labels = []
    curr_gallery_names = []
    with torch.no_grad():
        final_iter = tqdm(eval_trainloader, desc='Gallery Embedding Data...')
        for input_img, target, img_name in final_iter:
            curr_gallery_names.extend(list(img_name))
            gallery_labels.extend(target.numpy().tolist())
            emb = net(input_img.to(device),True)
            gallery_feature_colls.extend(emb.cpu().detach().numpy().tolist())
    
    curr_gallery_features = np.vstack(gallery_feature_colls).astype('float32')
    curr_gallery_features_cosine = normalize(curr_gallery_features, axis=1)
    curr_gallery_labels = np.hstack(gallery_labels).reshape(-1,1)
    # apply reindexing setting or not
    if (not reindex) and session_id > 0:
        gallery_features = np.vstack((curr_gallery_features,prev_gallery_features))
        gallery_features_cosine = np.vstack((curr_gallery_features_cosine,normalize(prev_gallery_features, axis=1)))
        gallery_labels = np.vstack((curr_gallery_labels,prev_gallery_labels))
        print("#gallery data until now: %d (#previous: %d)"%(len(gallery_features),len(prev_gallery_features)))
    else:
        gallery_features = curr_gallery_features
        gallery_features_cosine = curr_gallery_features_cosine
        gallery_labels = curr_gallery_labels
        print("#gallery data until now: %d (#previous: 0)"%(len(gallery_features)))
    #################### OBTAIN QUERY FEATURES ##############
    query_feature_colls = []
    query_labels = []
    with torch.no_grad():
        final_iter = tqdm(queryloader, desc='Query Embedding Data...')
        for input_img, target in final_iter:
            query_labels.extend(target.numpy().tolist())
            emb = net(input_img.to(device),True)
            query_feature_colls.extend(emb.cpu().detach().numpy().tolist())
    query_labels = np.hstack(query_labels).reshape(-1,1)
    query_features = np.vstack(query_feature_colls).astype('float32')
    query_features_cosine = normalize(query_features, axis=1)
    print("number of query data: %d"%(len(query_features)))

    #################### SETUP FAISS ####################
    faiss.omp_set_num_threads(kernels)
    torch.cuda.empty_cache()
    #################### START EVALUATION ####################

    """============ Compute Nearest Neighbours ==============="""
    if 'nearest_features' in requires:
        faiss_search_index  = faiss.IndexFlatL2(gallery_features.shape[-1])
        faiss_search_index.add(gallery_features)
        max_kval = np.max([int(x.split('@')[-1]) for x in metric_names if 'recall' in x])
        _, k_closest_points = faiss_search_index.search(query_features, int(max_kval))
        k_closest_classes   = gallery_labels.reshape(-1)[k_closest_points]
    if 'nearest_features_cosine' in requires:
        faiss_search_index  = faiss.IndexFlatIP(gallery_features_cosine.shape[-1])
        faiss_search_index.add(normalize(gallery_features_cosine,axis=1))
        max_kval = np.max([int(x.split('@')[-1]) for x in metric_names if 'recall' in x])
        _, k_closest_points_cosine = faiss_search_index.search(normalize(query_features_cosine,axis=1), int(max_kval))
        k_closest_classes_cosine   = gallery_labels.reshape(-1)[k_closest_points_cosine]
    record = {
        'e_recall_1' : list_of_metrics[0](query_labels,k_closest_classes),
        'e_recall_2' : list_of_metrics[1](query_labels,k_closest_classes),
        'e_recall_4' : list_of_metrics[2](query_labels,k_closest_classes),
        'c_recall_1' : list_of_metrics[3](query_labels,k_closest_classes_cosine),
        'c_recall_2' : list_of_metrics[4](query_labels,k_closest_classes_cosine),
        'c_recall_4' : list_of_metrics[5](query_labels,k_closest_classes_cosine),
    }
    print("c_recall@1: ",record['c_recall_1'], "c_recall@2: ",record['c_recall_2'],"c_recall@4: ",record['c_recall_4'])
    torch.cuda.empty_cache()

    if return_curr_gallery_names:
        return record, curr_gallery_features, curr_gallery_labels, curr_gallery_names
    else:
        return record, curr_gallery_features, curr_gallery_labels


def compute_acc(net, valloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.eval()
    correct = 0
    with torch.no_grad():
        for (data, target) in valloader:
            data, target = data.to(device), target.to(device)
            emb, output = net(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    total = len(valloader.dataset)
    acc = float(correct) / float(total)
    print('\nAccuracy: {}/{} ({:.6f}%)\n'.format(correct, total, acc))
    return acc
