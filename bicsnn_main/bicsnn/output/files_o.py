import torch

def output_log_traces(output, params_syn, logging, suffix, index_sample="last"):
    n_synapses = torch.numel(output['w'][:, :, -1])
    t_ms = output['s_out'].shape[1]
    p_ltp = 100 * int(torch.sum(output['w'][:, :, -1] > params_syn['thr_w'])) / n_synapses
    p_ltd = 100 * int(torch.sum(output['w'][:, :, -1] < params_syn['thr_w'])) / n_synapses
    v_post = torch.mean(torch.sum(output['s_out'], [2] ).to(float), [0,1]) / (t_ms / 1000)

    log_string = (
        "************************ \n Params:     v_post = {}       w_init = RAND synapses= {} \n P_LTP= {} \n P_LTD= {} \n **********************************************".format(
            v_post, n_synapses, p_ltp, p_ltd))
    logging.info(str(index_sample))
    logging.info(log_string)

    title = "Figure for {}  v_out = {}  w_init = RAND   P_LTP: {}   P_LTD: {}".format( suffix,
                                                                                                  v_post,
                                                                                                  p_ltp, p_ltd)
    #print(log_string)  # , flush=True)\
    #print("\n")
    #print("{} , {:.3f}  #vteach {} \n".format(int(v_post), p_ltp, params['v_teach']))
    #print("{} , {:.3f} #vteach {} \n".format(int(v_post), p_ltd, params['v_teach']))
    return title, int(v_post)