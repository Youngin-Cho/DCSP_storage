import vessl

from a2c.train import train
from a2c.cfg import get_cfg


if __name__ == "__main__":
    args = get_cfg()

    if bool(args.vessl):
        vessl.init(organization="snu-eng-dgx", project="DCSP_storage", hp=args)

    train(args)