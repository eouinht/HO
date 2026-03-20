* models.py - Định nghĩa dữ liệu
    Định nghĩa các class RU, DU, CU:
    * RU:
        * ru_id
        * du_id
        * cu_id 
        * (x,y)
        * total_prb 
        * total_ptx 
        * cell_type
    * DU:
        * du_id
        * cu_id 
        * capacity 
    * CU:
        * cu_id 
        * capacity

    * UE:
        * ue_id
        * serving_ru
        * du_id
        * cu_id 
        * (x,y)
        * sinr_db
        * rsrp_dbm
        * path_loss_db
        * bsr_bytes:
        * latencty_ms
        * mcs
        * cqi
        * ho_src
        * ho_dst 
        * slice_type
        * traffic_class 
        * payload_arrival_bytes
        * control_demmand
        * candidate_cells
        * air_metrics

    * Topology:
        * rus: danh sách các RU
        * dus: danh sách các DU 
        * cus: danh sách các CU
    * Phân loại HO:
        * NO_HO: Không HO
        * INTRA_DU_INTRA_CU: HO Khác RU cùng DU, CU
        * INTER_DU_INTRA_CU: HO khác RU, khác DU, cùng CU
        * INTER_CU: còn lại 

    * UEAction
        * target_ru
        * prb_alloc
        * ptx_alloc
        * du_alloc
        * cu_alloc
    * RewardWieghts
    * HOcost