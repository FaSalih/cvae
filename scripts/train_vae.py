from chemvae.train_vae import main_no_prop, main_property_run, config

# __name__ is set when executed from cli.
if __name__ == "__main__":
    # print("All config:", params)

    if config.do_prop_pred:
        main_property_run(config)
    else:
        print("no property pred.")
        main_no_prop(config)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--exp_file',
    #                     help='experiment file', default='exp.json')
    # parser.add_argument('-d', '--directory',
    #                     help='exp directory', default=None)
    # args = vars(parser.parse_args())
    # if args['directory'] is not None:

    # curdir = os.getcwd()
    # os.chdir(args['directory'])
    # args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    # os.chdir(curdir)
