###################################################
########### SCRIPTS FOR OUR EXPERIMENTS ###########
###################################################

# CIFAR-10
# CQ+Rand
python run.py --strategy=dtopk_random --dataset=cifar10 --model=resnet18 --initial_budget=1000 --budget=6000 --k=10 --n_rounds=10 --alpha=1 --cuda
# CQ+Ent
python run.py --strategy=dtopk_entropy --dataset=cifar10 --model=resnet18 --initial_budget=1000 --budget=6000 --k=10 --n_rounds=10 --alpha=1 --cuda
# CQ+BADGE
python run.py --strategy=dtopk_badge --dataset=cifar10 --model=resnet18 --initial_budget=1000 --budget=6000 --k=10 --n_rounds=10 --alpha=1 --cuda
# CSQ+Rand
python run.py --strategy=dtopk_random_conf --dataset=cifar10 --model=resnet18 --initial_budget=1000 --budget=6000 --k=0 --n_rounds=10 --alpha=1 --calibration_set_size 500 --cuda
# CSQ+Ent
python run.py --strategy=dtopk_entropy_conf --dataset=cifar10 --model=resnet18 --initial_budget=1000 --budget=6000 --k=0 --n_rounds=10 --alpha=1 --calibration_set_size 500 --cuda
# CSQ+Cost(Ent)
python run.py --strategy=dtopk_hybrid_entropy_conf --dataset=cifar10 --model=resnet18 --initial_budget=1000 --budget=6000 --k=0 --d=1.2 --n_rounds=10 --alpha=1 --calibration_set_size 500 --cuda
# CSQ+Cost(BADGE)
python run.py --strategy=dtopk_hybrid_badge_conf --dataset=cifar10 --model=resnet18 --initial_budget=1000 --budget=6000 --k=0 --d=1.0 --n_rounds=10 --alpha=1 --calibration_set_size 500 --cuda

# CIFAR-100
# CQ+Rand
python run.py --strategy=dtopk_random --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=100 --n_rounds=9 --alpha=1 --cuda
# CQ+Ent
python run.py --strategy=dtopk_entropy --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=100 --n_rounds=9 --alpha=1 --cuda
# CQ+BADGE
python run.py --strategy=dtopk_badge --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=100 --n_rounds=9 --alpha=1 --cuda
# CSQ+Rand
python run.py --strategy=dtopk_random_conf --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=0 --n_rounds=9 --alpha=1 --calibration_set_size 500 --cuda
# CSQ+Ent
python run.py --strategy=dtopk_entropy_conf --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=0 --n_rounds=9 --alpha=1 --calibration_set_size 500 --cuda
# CSQ+Cost(Ent)
python run.py --strategy=dtopk_hybrid_entropy_conf --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=0 --d=1.2 --n_rounds=9 --alpha=1 --calibration_set_size 500 --cuda
# CSQ+Cost(BADGE)
python run.py --strategy=dtopk_hybrid_badge_conf --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=0 --d=1.0 --n_rounds=9 --alpha=1 --calibration_set_size 500 --cuda

# CSQ+Ent with Top1
python run.py --strategy=dtopk_entropy --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=1 --n_rounds=9 --alpha=1 --cuda
# CSQ+Ent with Top10
python run.py --strategy=dtopk_entropy --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --k=10 --n_rounds=9 --alpha=1 --cuda
# Oracle with Entropy sampling
python run.py --strategy=ubdtopk_entropy --dataset=cifar100 --model=resnet18 --initial_budget=5000 --budget=6000 --n_rounds=9 --alpha=1 --cuda

# ImageNet64x64
# CQ+Rand
python run.py --strategy=dtopk_random --dataset=imagenet64 --model=wrn-36-5 --initial_budget=60000 --budget=60000 --n_rounds=16 --n_epochs=30 --k=1000 --alpha=1 --cuda --warmup_epochs=10 --warmup_lr=0.002 --push_warmup
# CQ+Ent
python run.py --strategy=dtopk_entropy --dataset=imagenet64 --model=wrn-36-5 --initial_budget=60000 --budget=60000 --n_rounds=16 --n_epochs=30 --k=1000 --alpha=1 --cuda --warmup_epochs=10 --warmup_lr=0.002 --push_warmup
# CSQ+Rand
python run.py --strategy=dtopk_random_conf --dataset=imagenet64 --model=wrn-36-5 --initial_budget=60000 --budget=60000 --n_rounds=16 --n_epochs=30 --k=0 --alpha=1 --calibration_set_size 5000 --cuda --warmup_epochs=10 --warmup_lr=0.002 --push_warmup
# CSQ+Cost(Ent)
python run.py --strategy=dtopk_hybrid_entropy_conf --dataset=imagenet64 --model=wrn-36-5 --initial_budget=60000 --budget=60000 --n_rounds=16 --n_epochs=30 --k=0 --d=1.2 --alpha=1 --calibration_set_size 5000 --cuda --warmup_epochs=10 --warmup_lr=0.002 --push_warmup

###################################################
###################### NOTES ######################
###################################################
# --k argument represents alpha in Eq. (2).
#     Setting this to 0 indicates using an optimized alpha for each AL round.
#     If this is not less than 1, it is used as a fixed candidate set size, 
#     which is for the conventional query (k=L) and top-1 and top-10 on CIFAR-100.
# --d is the hyperparameter for cost-efficient acquisition function. 
# --calibration_set_size is the size of the calibration set used for CSQ methods.
