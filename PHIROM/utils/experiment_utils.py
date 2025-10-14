import os


def get_path_and_name(args):
    dataset = args.dataset
    decoder_arch = args.decoder_arch
    if args.dino:
        method = "DINO"
    elif args.pinn:
        method = "PINN"
    else:
        method = "PHI-ROM"
    ad_ae = "AD" if args.autodecoder else "AE"
    normalize = "Normalize" if args.normalize else "Raw"
    path = os.path.join(dataset, ad_ae, decoder_arch, method)
    path = os.path.join(
        path,
        f"N={args.num_samples}_T={args.max_step}",
        f"LD={args.latent_dim}_Arch={decoder_arch}_Width={args.width}_Ac={args.activation}",
        f"NoArch={args.node_arch}_NoWidth={args.node_width}_NoAc={args.node_activation}",
    )
    if args.dino:
        path = os.path.join(
            path,
            f"{normalize}_Loss={args.loss}_Batch={args.batch_size}_"
            + f"LrD={args.learning_rate_decoder}_LrN={args.learning_rate_node}_"
            + f"LrL={args.learning_rate_latent}_DecayRate={args.decay_rate}_"
            + f"DecayEpochs={args.decay_steps}_GammaEpochs={args.gamma_epochs}",
        )
    else:
        path = os.path.join(
            path,
            f"{normalize}_Loss={args.loss}_Warmup={args.evolve_start}_"
            + f"Gamma={args.gamma}_Batch={args.batch_size}_"
            + f"LrD={args.learning_rate_decoder}_DecayRate={args.decay_rate}_"
            + f"DecayEpochs={args.decay_steps}",
        )

    if not args.adaptive:
        ode_steps = "fixed"
    else:
        ode_steps = args.max_ode_steps
    if args.dino:
        path = os.path.join(
            path,
            f"Epochs={args.epochs}_Seed={args.seed}_ODESolver={args.ode_solver}_ODESteps={ode_steps}",
        )
    else:
        path = os.path.join(
            path,
            f"Mode={args.node_training_mode}_Epochs={args.epochs}_"
            + f"Seed={args.seed}_ODESolver={args.ode_solver}_ODESteps={ode_steps}_Lambda={args.loss_lambda}",
        )
    name = f"Seed={args.seed}"
    if args.prefix:
        name = args.prefix + "_" + name
    return path, name
