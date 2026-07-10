# FTP-1 UniVTAC Finetune

`ftp1_univtac_finetune` is the UniVTAC fine-tuned release of FTP-1. It contains task-specific checkpoints initialized from the FTP-1 pretrained model and further trained on UniVTAC tasks.

This release currently includes 6 task checkpoints:

- `FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1`
- `FTP1_UniVTAC_insert_tube_expert_gsmall_ftp1`
- `FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1`
- `FTP1_UniVTAC_lift_can_expert_gsmall_ftp1`
- `FTP1_UniVTAC_pull_out_key_expert_gsmall_ftp1`
- `FTP1_UniVTAC_put_bottle_expert_gsmall_ftp1`

Each task directory contains a fine-tuned checkpoint at step `19999`.

## Notes

- Type: fine-tuned checkpoint collection
- Base model: `ftp1_pretrain_v0426_50kstep`
- Domain: UniVTAC
- Use: task-specific inference or further adaptation

## Links

- Project homepage: https://ftp1-policy.github.io/
- Open-source repository: https://github.com/michaelyuancb/ftp1-policy
- Pretrained checkpoints: https://huggingface.co/datasets/MJJJJ1064/ftp1_v0426_50kstep or https://www.modelscope.cn/models/michaelyuancb/ftp1_v0426_50kstep