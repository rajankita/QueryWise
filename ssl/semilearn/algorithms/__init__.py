# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core.utils import ALGORITHMS
name2alg = ALGORITHMS

def get_algorithm(args, net_builder, tb_log, logger, victim_model, labeled_set, val_set, test_loader):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm]( # name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger, 
            victim_model=victim_model,
            labeled_set=labeled_set, 
            val_set=val_set,
            test_loader=test_loader,
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.algorithm)}')



