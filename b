[33mcommit 221b1e4ec1242b33708ea89559dfc905e1f95242[m[33m ([m[1;36mHEAD[m[33m)[m
Author: Vahe Gharakhanyan <vaheg@meta.com>
Date:   Fri Jan 17 12:18:10 2025 -0500

    add dataset util tests (#953)
    
    * dataset util tests
    
    * config to pytest fixture
    
    * back to smaller lmdb
    
    * remove lmdb_dataset sample_property_metadata
    
    * ruff
    
    ---------
    
    Co-authored-by: Misko <misko@meta.com>
    Former-commit-id: a48a4eb9fc718dea06863919a36814d904bc6d11

[33mcommit 384768017bbf64db1703ece1a0bc13892f701857[m
Author: willis <wolearyc@gmail.com>
Date:   Fri Jan 17 00:05:20 2025 +0100

    Misc ASE db bugfixes (#965)
    
    * fixed bug in which row data was not fully copied during dataset split
    
    * fixed bug in which row data was not fully copied during dataset split
    
    Former-commit-id: 99f064a3d1b2e45842e136913e76e3427ed1f0ab

[33mcommit 8d1670622c2188a63a90093ff4536698ca35870c[m
Author: Ilias Chair <84736926+IliasChair@users.noreply.github.com>
Date:   Fri Jan 17 00:00:09 2025 +0100

    - replace all occurrences of `torch.cuda.amp.autocast(args...)` with `torch.autocast("cuda", args...)` (#930)
    
    - replace all occurrences of `torch.cuda.amp.GradScaler(args...)` with `torch.GradScaler("cuda", args...)`
    
    Co-authored-by: iliaschair <iliaschair14@gmail.com>
    Co-authored-by: Ray <7001989+rayg1234@users.noreply.github.com>
    Former-commit-id: fe6bb3248d782c629e539164d5bc7f69b9ee37d0

[33mcommit a955c23a3e192d147f8c4f3c7fa263a80078f04f[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Thu Jan 16 12:41:23 2025 -0800

    fix link typo (#971)
    
    
    
    Former-commit-id: cd3a217879dbc8783783028cba9aac45b5bb41a5

[33mcommit 20d6c414aafc63fbb0c70364b03898ab1d13418b[m
Author: Ray <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Jan 15 10:09:32 2025 -0800

    install hydra and torchtnt (#967)
    
    
    
    Former-commit-id: 8a0566224be172374eaeec4f1707521dae7a8da3

[33mcommit 3932b26b0aac5f83922abafb485f902ce029f98a[m
Author: Ankur Gupta <68587598+ankur-gupta-29@users.noreply.github.com>
Date:   Tue Jan 14 07:37:02 2025 +0530

    Update env.cpu.yml (#951)
    
    there is issue while install in cpu so i change cu121 to cpu
    
    Former-commit-id: b23dda40498697c24f6fb208a6dc3c2f7f0052a5

[33mcommit 60487bf4053260972d47993260f4cb252245173a[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Jan 13 18:06:57 2025 -0800

    fix train dataset links and correct val links and sizes (#963)
    
    
    
    Former-commit-id: 16fbdfa07e8e603c18c814289ecc00016ffdbff8

[33mcommit 1f1b2a12b6dbd3920ccf5b06331912c934b439d6[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Thu Jan 9 21:18:12 2025 -0800

    Bump ase from 3.23.0 to 3.24.0 (#952)
    
    Bumps [ase](https://gitlab.com/ase/ase) from 3.23.0 to 3.24.0.
    - [Changelog](https://gitlab.com/ase/ase/blob/master/CHANGELOG.rst)
    - [Commits](https://gitlab.com/ase/ase/compare/3.23.0...3.24.0)
    
    ---
    updated-dependencies:
    - dependency-name: ase
      dependency-type: direct:production
      update-type: version-update:semver-minor
    ...
    
    Signed-off-by: dependabot[bot] <support@github.com>
    Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
    Former-commit-id: 82baafcf88aeebeb5d17e6064849c45da387e3a0

[33mcommit 4a891d70a0865d537e3a2412149f03b5726e61a3[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Dec 20 14:50:23 2024 -0800

    Update README.md (#949)
    
    
    
    Former-commit-id: f7bcead25bd70c0772e473cb26ceabaa4f9f70eb

[33mcommit 93a574dcda2d2f29219992d208e636df6edd9c29[m
Author: Misko <misko@meta.com>
Date:   Thu Dec 19 11:31:04 2024 -0800

    Filter based on additional metadata fields (#948)
    
    * add in info fields
    
    add in info fields and limits
    
    * fix up datasetmetadata
    
    * lint
    
    * add some tests
    
    * replace DatasetMetadata with dict
    
    * remove datasetmetadata
    
    * remove datasetmetadata
    
    * remove info_fields, there is already a way to do this using r_data_keys
    
    Former-commit-id: 8b548913db0a5c597e1983cde430ba963a42521e

[33mcommit 32a5700af5fbee7444764078322732c4627b18ae[m
Author: Misko <misko@meta.com>
Date:   Wed Dec 18 12:18:39 2024 -0800

    add optional field to calculator to output only requested (#922)
    
    fix lint
    
    undo change to packages
    
    undo change to packages
    
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: 7a32302ac1427cf87dcbc2d69001d66345a2fcdb

[33mcommit 639f5bbcdfac2786a86b371462e53bbd1e410489[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Wed Dec 18 12:17:42 2024 -0800

    set tensor dtypes (#935)
    
    
    
    Former-commit-id: b183f41bc1c41747205fae8ab53c8e4242cac661

[33mcommit 4a34905a7a0102aba9d65fbf74aad26b6663b4b8[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Tue Dec 10 20:18:19 2024 +0000

    Bump codecov/codecov-action from 4 to 5 (#921)
    
    Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 4 to 5.
    - [Release notes](https://github.com/codecov/codecov-action/releases)
    - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
    - [Commits](https://github.com/codecov/codecov-action/compare/v4...v5)
    
    ---
    updated-dependencies:
    - dependency-name: codecov/codecov-action
      dependency-type: direct:production
      update-type: version-update:semver-major
    ...
    
    Signed-off-by: dependabot[bot] <support@github.com>
    Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: 5c4553d435da9e221a98058cf4ef7c3a1911869f

[33mcommit fcc4ee648c5803ac4efb811ac9e4b57c67cef293[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Dec 10 11:43:19 2024 -0800

    Add DCP support to main (#938)
    
    * temp switch to dcp instead
    
    * update
    
    * support both checkpoint styles
    
    * lint
    
    ---------
    
    Co-authored-by: rgao <rgao@meta>
    Former-commit-id: 40d9816799da78bb2478c7b402043da58cf5c631

[33mcommit f08985fdbcfc4fffed528add7abe9062f88e1962[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Dec 4 22:32:11 2024 -0800

    Add cuda set_device for local run (#931)
    
    * add set local device
    
    * fix test
    
    ---------
    
    Co-authored-by: rgao <rgao@meta>
    Former-commit-id: d436207467e3946a75b19954cd9d17b5fdabfc93

[33mcommit c076498b31bd18012ebf9e4986b9853f757a5763[m
Author: Misko <misko@meta.com>
Date:   Tue Dec 3 17:38:32 2024 -0800

    pin pytorch at 2_4_0 (#928)
    
    * pin pytorch at 2_4_0
    
    * also fix pypi package
    
    ---------
    
    Co-authored-by: misko user <misko@submit-1.fair-aws-h100-2.hpcaas>
    Former-commit-id: 0ffc1489ea464951df83b60af6100268969ab0ab

[33mcommit ec91ac3b0476b6f2d39375b8273a9e8e56e1434f[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Tue Dec 3 10:52:44 2024 -0800

    OptimizableBatch and stress relaxations (#718)
    
    * remove r_edges, radius, max_neigh and add deprecation warning
    
    * edit typing and dont use dicts as default
    
    * use super() and remove overkill deprecation warning
    
    * set implemented_properties from config
    
    * make determine step a method
    
    * allow calculator to operate on batches
    
    * only update if old config is used
    
    * reshape properties
    
    * no test classes in ase calculator
    
    * yaml load fix
    
    * use mappingproxy
    
    * expressive import
    
    * remove duplicated code
    
    * optimizable batch class for ase compatible batch relaxations
    
    * fix optimizable batch
    
    * optimizable goodies
    
    * apply force constraints
    
    * use optimizable batch instead and remove torchcalc
    
    * update ml relaxations to use optimizable batch correctly
    
    * force_consistent check for ASE compat
    
    * force_consistent check for ASE compat
    
    * check force_consistent
    
    * init docs in lbfgs
    
    * unitcellfilter for batch relaxations
    
    * ruff
    
    * UnitCellOptimizable as child class instead of filter
    
    * allow running unit cell relaxations
    
    * ruff
    
    * no grad in run_relaxations
    
    * make batched_dot and determine_step methods
    
    * imports
    
    * rename to optimizableunitcellbatch
    
    * allow passing energy and forces explicitly to batch to atoms
    
    * check convergence in optimizable and allow passing general results to atoms_from_batch
    
    * relaxation test
    
    * unit tests
    
    * move update mask to optimizable
    
    * use energy instead of y
    
    * all setting/getting positions and convergence in optimizable
    
    * more (unfinished) tests
    
    * backwards compatible test
    
    * minor fixes
    
    * code cleanup
    
    * add/fix tests
    
    * fix lbfgs
    
    * assert using norm
    
    * add eps to masked batches if using ASE optimizers
    
    * match iterations from previous implementation
    
    * use float64 for forces
    
    * float32
    
    * use energy_relaxed instead of y_relaxed
    
    * energy_relaxed and more explicit error msg
    
    * default to batch_size 1 if not set in config
    
    * keep float64 training
    
    * rename y_relaxed -> energy_relaxed
    
    * rm expcell batch
    
    * convenience commit from no_experimental_resolve
    
    * use numatoms tensor for cell factor
    
    * remove positions tests (wrapping atoms gives different results)
    
    * allow wrapping positions in batch to atoms
    
    * fix test
    
    * wrap_positions in batch_to_atoms
    
    * take a2g properties from model
    
    * test lbfgs traj writes
    
    * remove comments
    
    * use model generate graph
    
    * fix cell_factor
    
    * fix using model in ddp
    
    * fix r_edges in OCPcalculator
    
    * write initial and final structure if save_full is false
    
    * check unique atoms saved in trajectory
    
    * tighter tol
    
    * update ASE release comment
    
    * remove cumulative mask option
    
    * remove left over cumulative_mask
    
    * fix batching when sids as str
    
    * do not try to fetch energy and forces if no explicit results
    
    * accept Path objects
    
    * clean up setting defaults
    
    * expose ml_relax in relaxation
    
    * force set r_pbc True
    
    * make relax_opt optional
    
    * no ema on inference only
    
    * define ema none to avoid issues
    
    * lower force threshold to make sure test does not converge
    
    * clean up exception msg
    
    * allow strings in batch
    
    * remove device argument from lbfgs
    
    * minor cleanup
    
    * fix optimizable import
    
    * do not pass device in ml_relax
    
    * simplify enforce max neighbors
    
    * fix tests (still not testing stress)
    
    * pin sphinx autoapi
    
    * typo in version
    
    ---------
    
    Co-authored-by: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
    Co-authored-by: Zack Ulissi <zulissi@meta.com>
    Former-commit-id: 69f13a70241995cdbfab66b8ce1d1459aa10c229

[33mcommit a8c448bb19a13e0351293e848bac5fe3009416f7[m
Author: Xiang <31351668+kyonofx@users.noreply.github.com>
Date:   Tue Nov 26 15:13:10 2024 -0800

    EquiformerV2 + DeNS model and trainer (#880)
    
    * add density metrics
    
    * update trainer & loss
    
    * interleave atoms in loss
    
    * fix call to keys
    
    * add rmse to evaluation metrics
    
    * fix linting.
    
    * per_atom_loss fix
    
    * fix test
    
    * Equiformer DeNS model and trainer
    
    * fix linting.
    
    * lint
    
    * lint again
    
    * add type hints
    
    * empty cuda cache and remove db closing
    
    * type hints
    
    * add missing args to docstring
    
    * add return type hints
    
    * rename dens heads
    
    * move use_densoising to heads
    
    * abstract denoising targets
    
    * update omat24 dens config
    
    * fix imports
    
    * fix trainer
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Co-authored-by: Brandon <bmwood@meta.com>
    Former-commit-id: d1a8dca088394d4cd8513143240822dd6610a4ae

[33mcommit 1e275eb5a024c94adc2b66a1b1a87efbc604b08e[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Mon Nov 25 17:22:10 2024 -0800

    Add wandb logger init to hydra runners (#894)
    
    * add wandb logger init to hydra runners
    
    * update to reading dict vars
    
    * update to reading dict vars
    
    * get rid of finally clause
    
    * move logger init
    
    * add deprecation comment
    
    * Revert "add deprecation comment"
    
    This reverts commit 3a0f75247a80bbcb10c56ab7e9e7e598f4a92f5f [formerly a53caa73a177718b402b35ed1df790affe3430ce].
    
    Former-commit-id: 1ade44cbbece31076ad668cc7821705d439adce0

[33mcommit 4572311d9c4553583bbddd6ba7568a5d7fb1745e[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Nov 25 16:54:44 2024 -0800

    update omat24 configs for new loss (#927)
    
    * update omat24 configs for new loss
    
    * fix typo
    
    Former-commit-id: bf97537a34227e52df83250b01ceeac08a8c3a4a

[33mcommit 1ba8f17b39842500f07856f67d9288c424983a1f[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Nov 25 15:25:33 2024 -0800

    Update ODAC model_checkpoints.md (#893)
    
    
    
    Former-commit-id: f53809b9ec70c6589980a5e5df8103d5360d75e2

[33mcommit 00b689a6f06bb8c74033e6c6716b77a1f09c1bf2[m
Author: Ilias Chair <84736926+IliasChair@users.noreply.github.com>
Date:   Thu Nov 21 23:30:05 2024 +0100

    remove duplicate header [tool.hatch.build] from fairchem-data-om pyproject.toml file (#917)
    
    Co-authored-by: iliaschair <iliaschair14@gmail.com>
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: 080993d3fe587c6b2b2290e25e78129b58a484f3

[33mcommit ce54d4606edb6c3b3e70b0af901593d63424d030[m
Author: Ilias Chair <84736926+IliasChair@users.noreply.github.com>
Date:   Thu Nov 21 18:45:42 2024 +0100

    add ema to BaseTrainer init (#916)
    
    Co-authored-by: iliaschair <iliaschair14@gmail.com>
    Former-commit-id: 01a58477663146156725584d64bf422d9d57bbee

[33mcommit fb91a0818945077fe0284c3c210a2c2f869a645f[m
Author: Xiang <31351668+kyonofx@users.noreply.github.com>
Date:   Tue Nov 19 15:30:08 2024 -0800

    replication use .item(); fix hydra_cli (#920)
    
    * hydra accomodation for compilatable models
    
    * minor changes for fxdev
    
    * modify hydra cli
    
    * fix linting and remove ununsed
    
    * reset trainer and reformat utils
    
    * fix linting
    
    * fix hydra cli
    
    * recover hydra cli config
    
    Former-commit-id: e1b236c303c1cda63987caac723f001a4a755f98

[33mcommit 051327bc185e5b858448e0c14c48e1e7e4dc2d3b[m
Author: Misko <misko@meta.com>
Date:   Tue Nov 19 12:24:07 2024 -0800

    Create a issue template for bugs (#912)
    
    * borrowing from pymatgen a bug tracking template
    
    * rename
    
    * update bug report yaml
    
    * inspired from QUACC add in misc intake form
    
    Former-commit-id: 547cb30200a128bdc68b5c6002a7927908df456d

[33mcommit f94762d83a2f24c85b8a497c5730a0c90e5c0ad6[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Mon Nov 18 22:48:40 2024 -0600

    OCx24 Release (#915)
    
    * initial ocx scripts
    
    * adding documentation
    
    * adding summary figure
    
    * final updates to ml inference needs documentation and data
    
    * adding docs + small tweaks
    
    * adding all data
    
    * ocx updates
    
    * add xrf data
    
    * create data readme
    
    * simplify readme
    
    * add supporting data
    
    * experimental metadata
    
    * Update README.md
    
    fixing incorrect link
    
    * adding links to the analysis readme
    
    * rip a space
    
    * updating the md file for the gitbook page
    
    * adding XRD html tar files
    
    * Create README.md for the experimental data folder
    
    * Create README.md for the processed data folder
    
    * fix paths
    
    * Update experimental data readme to include supporting data info
    
    * readme edits and adding cod lookup file
    
    * fix link
    
    * Add files via upload
    
    * small readme tweaks
    
    * add abbreviations
    
    * adding last file and paper link
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: b4bbc6e56835c785e2eca88d0588590197e527a7

[33mcommit 30feb6ddd282b25f2feebb36cb0cb7f2821eb991[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Nov 12 18:07:01 2024 -0800

    Add utils for logging weight tables and tensor stats (#907)
    
    * log stuff for debugging escn
    
    * add log helper
    
    * log table for weights
    
    * move weight table to utils
    
    * undo
    
    * typo
    
    * tmp add headname
    
    * add shape
    
    * restore base trainer
    
    * remove some things
    
    Former-commit-id: e450506a3315a854c96f71ae1824fa0eeffdc2c2

[33mcommit 1285e4b4e60775e7e71cdcc86975391f26db5397[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Nov 11 11:44:04 2024 -0800

    sort file paths (#904)
    
    
    
    Former-commit-id: 5c2c2c852922a01138f10ed34a19aad02a958aa7

[33mcommit f5633c2f974f60229b06c5e5b9dbb20a30566271[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Nov 11 11:43:58 2024 -0800

    Omat24 doc fix (#908)
    
    * update citation
    
    * fix typo
    
    Former-commit-id: bd11581a036dd942983f64840da21700b1d4024d

[33mcommit f4e05437179cdf1ed459b021f211cc5fb163e32d[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Nov 8 11:57:44 2024 -0800

    Rename hydra heads (#903)
    
    * refactor eqV2 heads
    
    * refactor init_weights
    
    * add output_name attribute to eqV2 heads
    
    * fix deprecated registry names
    
    * remove debug breakpoint
    
    * add default name for rank2 head
    
    Former-commit-id: 4ea0231431c29fed1d61b888ed8ddbf957e5bf13

[33mcommit 851da377182a68495deb56d0d76b382297b2b580[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Nov 4 20:03:02 2024 -0600

    add omat24 docs (#881)
    
    * add omat24 docs
    
    * add links to OMat24
    
    * add to index and toc
    
    * add training configs
    
    * add HF links
    
    * rm mp only ft
    
    * add comment to request access via HF
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: e0851b039b72f201ebb95451cd858495fafda556

[33mcommit f490f6ce75d03ee905c6610998f083840c286f0c[m
Author: Brandon Wood <bmwood@meta.com>
Date:   Wed Oct 30 10:01:35 2024 -0700

    Updated loss and eval metrics (#896)
    
    * add density metrics
    
    * update trainer & loss
    
    * interleave atoms in loss
    
    * fix call to keys
    
    * add rmse to evaluation metrics
    
    * fix linting.
    
    * updated loss module with tests
    
    * more tests and updated eval names
    
    * failing test update
    
    * renaming norm loss p2 -> l2
    
    * minor changes based on comments
    
    * added inline comment
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Xiang Fu <xiangfu@meta.com>
    Former-commit-id: 0486bfc7b8c7d9e233a9685155bfefff655df5d4

[33mcommit cb77fe9ade691e7541c9b7314377dd4ba438eabf[m
Author: Misko <misko@meta.com>
Date:   Tue Oct 29 11:44:03 2024 -0700

    add hydra freeze backbone option (#898)
    
    
    
    Former-commit-id: ad31c33175bb05f59fe3cbcca636ae0c721befa7

[33mcommit 7469108cb4b9a0d1ec0ad3682d96f7b586a69a2d[m
Author: Misko <misko@meta.com>
Date:   Thu Oct 24 16:31:30 2024 -0700

    make it work with old checkpoints (#856)
    
    
    
    Former-commit-id: 4a821ce4d0156c41c7e71b3037b7b05d8e9da1f0

[33mcommit 76a88fa4c98efba4e64ca431aa6112ea63ce5131[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Oct 23 15:23:28 2024 -0500

    add per adsorbate download links (#891)
    
    
    
    Former-commit-id: c0f3c99e9bc62a41a488d959de7c3e52f4b7eea2

[33mcommit 88d0f2fce44cf4679948ba9118586d0cbc54cf52[m
Author: Misko <misko@meta.com>
Date:   Wed Oct 23 12:19:18 2024 -0700

    Read the logger logging level from environment variable (#874)
    
    * read logging level from env
    
    * fix lint
    
    Former-commit-id: f4579551969b2b566943e8895076a62fc4374a42

[33mcommit c945a931729112af41e7b7c654b57eca73bcddeb[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Wed Oct 23 10:52:01 2024 -0700

    Bump torch-geometric from 2.6.0 to 2.6.1 (#870)
    
    Bumps [torch-geometric](https://github.com/pyg-team/pytorch_geometric) from 2.6.0 to 2.6.1.
    - [Release notes](https://github.com/pyg-team/pytorch_geometric/releases)
    - [Changelog](https://github.com/pyg-team/pytorch_geometric/blob/master/CHANGELOG.md)
    - [Commits](https://github.com/pyg-team/pytorch_geometric/compare/2.6.0...2.6.1)
    
    ---
    updated-dependencies:
    - dependency-name: torch-geometric
      dependency-type: direct:production
      update-type: version-update:semver-patch
    ...
    
    Signed-off-by: dependabot[bot] <support@github.com>
    Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
    Former-commit-id: 322a6ab53f44c7698a94c5ddcf0a6ae980bf12ae

[33mcommit 1f5946dfe204d1988c775c6ab10c0c8fa106ce6f[m
Author: Misko <misko@meta.com>
Date:   Wed Oct 23 09:27:31 2024 -0700

    remove destroy_process_group() from finally wrapper as it can hang (#884)
    
    * remove cleanup from finally as it can hang
    
    * we need destroy_group for tests; at least dump error message and try to exit gracefully
    
    * patch tests
    
    * fix linting :(
    
    Former-commit-id: 362fca63768f742b95bb4c164218b99db07b3f07

[33mcommit 74b4dad21b8b066963d5df8efc8d7df36995956b[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Tue Oct 22 12:09:55 2024 -0400

    adding tutorial (#882)
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: a1b6d61e70ae4181b134cc1e2eff160fbbdcc478

[33mcommit 01b36c29cb1aa3578b0db8b9fa6c3e51238a9cb1[m
Author: Misko <misko@meta.com>
Date:   Sun Oct 20 19:51:42 2024 -0400

    Add script to make release from fine tuned hydra (#875)
    
    * add script to convert hydra checkpoint to release
    
    * add tests
    
    Former-commit-id: 1404cd34a4a83620e3cd7a34744fbf52c8a7c55a

[33mcommit 9b82b59fb6dd37553a0dcdb3c603564c2715f2c3[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Fri Oct 18 15:48:40 2024 -0700

    Add hydra entrypoint (#867)
    
    * format
    
    * working relaxations on local, add submitit
    
    * small fixes
    
    * set logging
    
    * add reqs
    
    * add tests
    
    * add tests
    
    * move hydra package
    
    * revert some changes to predict
    
    * cpu tests
    
    * add cleanup
    
    * typo
    
    * add check for dist initialized
    
    * update test
    
    * wrap feature flag for local mode as well
    
    Former-commit-id: 9cd075a0ad78aa4241e311a2d4ab23aa08ebdc18

[33mcommit 581080e498c8a53325edd64300a390453cf8dbde[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Mon Oct 14 12:55:08 2024 -0500

    adding link to full dataset (#869)
    
    * adding link to full dataset
    
    * removing whitespace
    
    Former-commit-id: 8c38314e9f0513359de284e0d49def3f8776d078

[33mcommit ab677fd5a25ab86fdafbaf361f20e598dccd7a8f[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Oct 9 12:08:47 2024 -0700

    Add singleton logger (#873)
    
    * add singleton wandb logger
    
    * add trainer update
    
    * update comment
    
    * update comment
    
    Former-commit-id: 6829e8fb5ceee48375dd58b470db3cb541fef048

[33mcommit d3048e9d16189d700e89ea726e9cc32b0fb2b134[m
Author: Misko <misko@meta.com>
Date:   Tue Oct 1 14:05:18 2024 -0700

    Add tests for hydra gemnet OC scaling factor generation and loading; Raise error on fail to load scaling factors (#831)
    
    * raise on gemnet oc fail to load scaling; add scaling tests for hydra oc aswell
    
    * remove assert
    
    * always raise error if scalings factors cannot be loaded
    
    * add debug to figure out why github ci is failing
    
    * rename for always ddp
    
    * remove loading scaling factors for num blocks >3
    
    Former-commit-id: a336ca20f34f1338631f007963d936a00671aa74

[33mcommit 4a7ddfe209ff89ab90630c42f65a8fea55b7e296[m
Author: Misko <misko@meta.com>
Date:   Tue Sep 24 11:21:16 2024 -0700

    add act checkpointing to escn (#852)
    
    * add act checkpointing to escn
    
    * remove re-entrant from non checkpoint
    
    * assign output correctly
    
    * remove escneqv2heads
    
    * fix up comment
    
    Former-commit-id: 7edcd7d274e4492091ba2f31f14ca55117a5f059

[33mcommit ae2d168fba43b4993c15db2e94398dfce90d6fa5[m
Author: Misko <misko@meta.com>
Date:   Fri Sep 20 13:55:06 2024 -0700

    Match escn_exportable with escn main (#866)
    
    * fix l/m; make rescaling optional
    
    * remove mapping reduced from tests, hoping we wont need to register buffer
    
    Former-commit-id: 7d5207d611f87f979e09794cad5d6fc90f55f526

[33mcommit 7a99a46210920fc7a15352f4f424bc7d48001fe3[m
Author: Misko <misko@meta.com>
Date:   Thu Sep 19 14:38:28 2024 -0700

    throw error when metadata is non integer (#865)
    
    
    
    Former-commit-id: 504b698da4d5e60eb906c483cb20ec5807fd4532

[33mcommit 3fa1eaa97df9534517e24ecdba639d5af7c24aa1[m
Author: Brandon Wood <bmwood@meta.com>
Date:   Thu Sep 19 13:29:38 2024 -0700

    change logic to catch missing outputs (#859)
    
    * change logic to catch missing outputs
    
    * minor update, changed error type and added out list
    
    * fix decomposition in output logic
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Former-commit-id: 4d670fb997b0ba123152f2db9d12a40c71a9ab95

[33mcommit 957876535576259a3c29c58bd71caf21a2ed5895[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Wed Sep 18 16:07:23 2024 -0700

    update CI tests to py 3.12 (#858)
    
    * update CI tests to py 3.12
    
    * bump numpy to work with py 3.12
    
    * bump numpy everywhere to work with py 3.12
    
    Former-commit-id: ebf14d0c19c5b762d9134661c4015e2fe3db8533

[33mcommit e43453cd7ab9229e2c124061266d79d25047150c[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Sep 18 10:04:53 2024 -0700

    add export link (#864)
    
    
    
    Former-commit-id: 2714d61abdbcd55736418347602104f302223b1c

[33mcommit 353642dcbaea15199015afed1b89678ad5364fd3[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Wed Sep 18 05:36:55 2024 +0000

    Bump torch-geometric from 2.3.0 to 2.6.0 (#854)
    
    Bumps [torch-geometric](https://github.com/pyg-team/pytorch_geometric) from 2.3.0 to 2.6.0.
    - [Release notes](https://github.com/pyg-team/pytorch_geometric/releases)
    - [Changelog](https://github.com/pyg-team/pytorch_geometric/blob/master/CHANGELOG.md)
    - [Commits](https://github.com/pyg-team/pytorch_geometric/compare/2.3.0...2.6.0)
    
    ---
    updated-dependencies:
    - dependency-name: torch-geometric
      dependency-type: direct:production
      update-type: version-update:semver-minor
    ...
    
    Signed-off-by: dependabot[bot] <support@github.com>
    Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
    Former-commit-id: cd20e08a483214994a48b9a41c17a7635a894c1e

[33mcommit 8338f5c77a16bcc3d360d20c2491350f368a3af1[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Sep 17 15:38:57 2024 -0700

    fix bug where trainer state not being loaded (#863)
    
    * fix bug where trainer state not being loaded
    
    * lint
    
    * fix ase
    
    Former-commit-id: 4176e63ea272fc0867b6573904861ffd9e869b33

[33mcommit c47ba490465de3319088010a95b902936c81453d[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Sep 17 12:25:39 2024 -0700

    ESCN export II (#848)
    
    * temp fix to train mptraj
    
    * add compile option
    
    * move jd to init
    
    * fix dynamic export
    
    * add value testing for export
    
    * update test
    
    * lint
    
    * update comment
    
    * update forward code
    
    * reraise error
    
    * wrap escn
    
    * revert packages
    
    * format
    
    Former-commit-id: becd99da79beed9d72e119fcefcc31cf5410131b

[33mcommit 91139f7127bab7e113eef8d6737594950eb8cfd0[m
Author: Kyle Michel <59006240+kjmichel@users.noreply.github.com>
Date:   Tue Sep 17 11:57:12 2024 -0700

    ocpapi dependencies / requests -> 2.32.3 (#862)
    
    
    
    Former-commit-id: fb8efc729c015f65932ad55c427343c332107e6d

[33mcommit f41b231aa891a6a48c7eb3a1d9066dae4acebd81[m
Author: Misko <misko@meta.com>
Date:   Tue Sep 17 09:42:54 2024 -0700

    enable ruff formatter (#853)
    
    
    
    Former-commit-id: 999c51bb1c4e4842c2c27e3e15e6f1a46e650b64

[33mcommit 623a6e6f5c63930f4ccae3715b8cb2f7d5e60c6f[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Sep 16 19:08:09 2024 -0700

    bump torch in pyproject.toml (#857)
    
    * bump torch in pyproject.toml
    
    * also cattsunami
    
    Former-commit-id: 8e0daaf231eb56a22e09e4afcb3a17275ba828b4

[33mcommit 8e75a19a89b456a792ad4ff369888b716296d09a[m
Author: Misko <misko@meta.com>
Date:   Mon Sep 16 14:00:11 2024 -0700

    Bump torch2.4.1 and pyg (#845)
    
    * bump torch and pyg
    
    * fix exported_prog call according to torch error message
    
    * add .module()
    
    * missed one last one
    
    * try with torch 2.4.1
    
    * update yml configs
    
    Former-commit-id: e116d5c2088ab1012f60afa1753b559341131622

[33mcommit bbe12f5dfeb38aa64cc027e960e75db6ff073c4a[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Sep 13 16:40:52 2024 -0700

    do not shadow built-in property (#841)
    
    
    
    Former-commit-id: fbe79a27984fc7c30d8838f02a994bc301222213

[33mcommit bce368652e4bbab969af19e5b6aef8d01da4f216[m
Author: Misko <misko@meta.com>
Date:   Fri Sep 13 14:35:38 2024 -0700

    add script to port old equiv2 checkpoint+yaml to hydra version (#846)
    
    * add script to port old equiv2 checkpoint+yaml to hydra version
    
    * fix up comments
    
    * lint
    
    * move script and add forces to test
    
    Former-commit-id: 64e7c4e138d4d1c91d0b7e549a521c48ba880168

[33mcommit 3709456b0caf52ec4bee691e24c2d28134037b13[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Sep 13 14:04:47 2024 -0700

    Add inference_only option to trainers (#850)
    
    * deprecate ocp-collater
    
    * add type hints to trainers
    
    * fix deprecation warning
    
    * add inference_only key to avoid loading unnecessary things
    
    * no need to remove relax dataset anymore
    
    * fix typo
    
    * remove deprecated decorator
    
    * fix normalizer loading
    
    * fix typo
    
    Former-commit-id: fa10c0b0cd1636984ed7e06e8a15eb3a9458c944

[33mcommit 70917cdf2ceffbdb530e234544f6e48761d51dbf[m
Author: Daniel Levine <levineds@meta.com>
Date:   Fri Sep 13 09:44:20 2024 -0700

    Remove large file from repo prior to git history purge (#844)
    
    * Remove large file from repo prior to git history purge
    
    * add missing download in the docs
    
    Former-commit-id: b9f4fdeb28ce8f1c40acc0fc4870cea277b846a9

[33mcommit 8030ba5e9b9a81c2710890af6c8faf3027e21216[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Sep 10 14:05:45 2024 -0700

    Delete distributed option - Always use DDP (#833)
    
    * delete distributed option
    
    * update
    
    * fix test
    
    * fix test
    
    * more cleanup
    
    * update
    
    * update
    
    * fix test
    
    * fix test
    
    * update
    
    * fix tests and slurm runs
    
    * update
    
    * typo
    
    * revert 2 files
    
    * test build docs
    
    * test build docs
    
    * fix book
    
    * fix book
    
    * fix book
    
    Former-commit-id: 5193d01439e21aceb172c2b2700ed50c98de6117

[33mcommit 65909a898ecabedeb6f8b0af61535e107a86bf4a[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Sep 10 13:16:33 2024 -0700

    Torch compile + export escn (#826)
    
    * update
    
    * update
    
    * export so2
    
    * move mappingReduced to member
    
    * compile works, guard failures still
    
    * escn so2 exports
    
    * add gpu test
    
    * pass cuda test
    
    * layer block
    
    * switch to separate export file
    
    * message block fails export due to SO3Rotation input
    
    * message block compiles and exports
    
    * layer block compiles and exports
    
    * remove most of lmax_list and mmax_list
    
    * remove eqv2 stuff from this branch
    
    * compile works
    
    * update
    
    * remove some files from main
    
    * lint
    
    * ruff
    
    * lint
    
    * revert base trainer changes
    
    * cleanup a2g
    
    * address comments
    
    * cleanup
    
    Former-commit-id: 6e94cc295bf29c110356f06110a63772428f7f2c

[33mcommit d076f8cddcc5b63e437d0b11c0f7cfeeedd6afc5[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Sep 10 11:26:40 2024 -0500

    explicitly define ispin for oc20 bulks (#840)
    
    * explicitly define ispin
    
    * explicit ispin adslabs
    
    Former-commit-id: 164bd0ed8a2abc41a8de45b6ad14d27c811d33df

[33mcommit 623c3b8cf241e8249f864a23747ff0f640202d18[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Sep 9 11:01:08 2024 -0700

    sort filepaths (#837)
    
    
    
    Former-commit-id: 4cf36c31c37f15d3a73f0e34484499f07c9553ac

[33mcommit 75a0a2f30f5990d8b0a4377731f8a4658454dff1[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 5 20:07:00 2024 -0500

    add docstring. Add support for different pseudopotential path (#832)
    
    * add docstring. Add support for different pseudopotential path
    
    * update ediff
    
    * fix minor typo
    
    Former-commit-id: ba035896e98be87d4646b5e297a5da0397dd24cd

[33mcommit d27aae132639dfaa77f237113601e0d8d60e2884[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 5 09:53:49 2024 -0500

    add tags, fix min_ab slab (#829)
    
    * add tags, fix min_ab slab
    
    * add pp setups support
    
    Former-commit-id: 4b04f63eaa5a1f6cf20f501029a9eaba1bdfe22f

[33mcommit 3f8a925b7e3f66785f93ea8a2c3f9af0174ee9c2[m
Author: Brandon Wood <bmwood@meta.com>
Date:   Wed Sep 4 17:41:04 2024 -0700

    update to gemnet-oc hydra force head to work with amp (#825)
    
    * updated gemnet hydra force head to work with amp
    
    * remove use_amp
    
    ---------
    
    Co-authored-by: Misko <misko@meta.com>
    Former-commit-id: 041e96fed9db099db18fdd0bc61f54e91c6a5050

[33mcommit ac2823aaa31215fd05f87f2331e157a71c2f0a05[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Sep 4 17:16:02 2024 -0700

    add mean e to escn energy head (#828)
    
    
    
    Former-commit-id: fd149b312f8d263c847627b44ce1a9c809034875

[33mcommit 8b7da3b21c277eba0a3033a432f4c8ff2868f948[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Wed Sep 4 15:06:37 2024 -0700

    fix loading normalizers from checkpoint with OTF (#824)
    
    
    
    Former-commit-id: ff9dd798feb04246ebec7743e67e4dbdd0a549b6

[33mcommit d9d89e4778341d4486800c3c96796096677eb136[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Sep 4 10:35:49 2024 -0700

    add run group (#827)
    
    
    
    Former-commit-id: 857ecdbd2f69ff41068fb101ff380ba138337790

[33mcommit 75fc5cc7d7cf1c88bc8d1934b3cb045c3c3c2fe6[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Thu Aug 22 10:52:34 2024 -0700

    add option to apply mean to energy (#818)
    
    * add option to apply mean to energy
    
    * lint
    
    * change name to reduce
    
    * update
    
    Former-commit-id: a23ca83b8c2d60fa1a4e45646870819930686014

[33mcommit 2028cea1ba88494e2aae455213f50545dd6db7cd[m
Author: Misko <misko@meta.com>
Date:   Wed Aug 21 19:46:49 2024 -0700

    fix gemnet scaling factors fit.py and add a test (#819)
    
    * fix gemnet fit and add test
    
    * only save factors, not the whole model!
    
    Former-commit-id: a2f81b8c597f1ab6d26649d1bddb691833f15265

[33mcommit f628b9681f48cdb50fa85313812427859b0b5587[m
Author: Misko <misko@meta.com>
Date:   Wed Aug 21 17:07:47 2024 -0700

    Add check to max num atoms (#817)
    
    * add assert for max_num_atoms
    
    * add test to make sure we are properly checking for max_num_elements
    
    * fix post merge
    
    Former-commit-id: 6943df40f9479669cb3a73cf62203021a833b518

[33mcommit 45e64d652459427d1bbba47d7feda4fb18ddc57a[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Aug 21 16:04:19 2024 -0700

    update to use abs run_dir paths by default (#820)
    
    
    
    Former-commit-id: 863f2fc0062756beabff4f92a8829e170edeb896

[33mcommit 9df39fbf5055a28d658244018f1c0fa51488188f[m
Author: Misko <misko@meta.com>
Date:   Tue Aug 20 19:58:22 2024 -0700

    FM-v4 branch into main (#752)
    
    * Update BalancedBatchSampler to use datasets' `data_sizes` method
    Replace BalancedBatchSampler's `force_balancing` and `throw_on_error` parameters with `on_error`
    
    * Remove python 3.10 syntax
    
    * Documentation
    
    * Added set_epoch method
    
    * Format
    
    * Changed "resolved dataset" message to be a debug log to reduce log spam
    
    * Minor changes to support multitask
    
    * add in pickle data set; add in stat functions for combining mean and variance
    
    * checksums for equiformer
    
    * detach compute metrics and add checksum function for linear layer
    
    * change name to dataset_configs
    
    * add seed option
    
    * remove pickle dataset
    
    * remove pickle dataset
    
    * add experimental datatransform to ase_dataset
    
    * clean up batchsampler and tests
    
    * base dataset class
    
    * move lin_ref to base dataset
    
    * inherit basedataset for ase dataset
    
    * filter indices prop
    
    * updated import for ase dataset
    
    * added create_dataset fn
    
    * yaml load fix
    
    * create dataset function instead of filtering in base
    
    * remove filtered_indices
    
    * make create_dataset and LMDBDatabase importable from datasets
    
    * create_dataset cleanup
    
    * test create_dataset
    
    * use metadata.natoms directly and add it to subset
    
    * use self.indices to handle shard
    
    * rename _data_sizes
    
    * fix Subset of metadata
    
    * fix up to be mergeable
    
    * merge in monorepo
    
    * small fix for import and keyerror
    
    * minor change to metadata, added full path option
    
    * import updates
    
    * minor fix to base dataset
    
    * skip force_balance and seed
    
    * adding get_metadata to base_dataset
    
    * implement get_metadata for datasets; add tests for max_atoms and balanced partitioning
    
    * a[:len(a)+1] does not throw error, change to check for this
    
    * bug fix for base_dataset
    
    * max atoms branch
    
    * fix typo
    
    * do pbc per system
    
    * add option to use single system pbc
    
    * add multiple mapping
    
    * lint and github workflow fixes
    
    * track parent checkpoint for logger grouping
    
    * add generator to basedataset
    
    * check path relative to yaml file
    
    * add load and exit flag to base_trainer
    
    * add in merge mean and std code to utils
    
    * add log when passing through mean or computing; check other paths for includes
    
    * add qos flag
    
    * use slurm_qos instead of qos
    
    * fix includes
    
    * fix set init
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * remove files with diff whitespace
    
    * add resolution flag to escn
    
    * try to revert oxides
    
    * revert typing
    
    * remove white space
    
    * extra line never reached
    
    * move out of fmv4 into dev
    
    * move avg num nodes
    
    * optional import from experimental
    
    * fix lint
    
    * add comments, refactor common trainer args in a single dictionary
    
    * add comments, refactor common trainer args in a single dictionary
    
    * remove parent
    
    ---------
    
    Co-authored-by: Nima Shoghi <nimashoghi@gmail.com>
    Co-authored-by: Nima Shoghi <nimashoghi@fb.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Brandon <bmwood@meta.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Co-authored-by: Ray Gao <rgao@meta.com>
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: ddca24db0276ec6734b0c342ac78a84b9b15b218

[33mcommit 72eff56d2ad882af207df670d45507d7aa63e1e3[m
Author: Misko <misko@meta.com>
Date:   Tue Aug 20 17:03:24 2024 -0700

    refactor and deprecate old equiformerv2 (#812)
    
    * refactor and deprecate old equiformerv2
    
    * remove equiv2_backbone_and_heads
    
    * lint fixes
    
    * remove backbone and heads model
    
    * fix merge
    
    * split up tests
    
    * update
    
    * add in missing file
    
    Former-commit-id: badab4a26e36966a977e3d5889556fe22789c2d1

[33mcommit a1b933a2949a4850df4028bea74e404ba71c6a75[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Mon Aug 19 16:16:05 2024 -0700

    Fuse all hydras (#814)
    
    * make hydra compat with multitask
    
    * remove finetune hydra
    
    * fix tests
    
    * update comment
    
    * update logic slightly
    
    * ruff
    
    * fix test
    
    * add map location
    
    * ruff
    
    * fix tests
    
    * get device from input in hydra
    
    * update ocp_hydra_example.yml
    
    * add logging
    
    * update
    
    Former-commit-id: d84f8138b166dcda3e7d29da5fba303bd9299be1

[33mcommit 867c3601aae4131af4fe728501d417e31137b821[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Aug 16 18:02:33 2024 -0700

    Rank2 tensor head (#792)
    
    * rank 2 tensor head
    
    * fix rank2 head and add to e2e test
    
    * small fixes
    
    * keep hydra graphmixin
    
    * add rank2 head tests
    
    * test fixes
    
    * fix tests; move init_weight out of equiformer; add amp property to heads+hydra
    
    * add amp to heads and hydra
    
    * fix import
    
    * update snapshot; fix test seed and change tolerance
    
    ---------
    
    Co-authored-by: Misko <misko@meta.com>
    Former-commit-id: 93afd03b796fd354cad5acd9a8676bee81a4a045

[33mcommit bfd0e8a3f1ee9af0f9a17b782355dc1b8a9fc4a2[m
Author: Misko <misko@meta.com>
Date:   Thu Aug 15 17:35:01 2024 -0700

    Explicitly initialize weights even if initialization method is "uniform" (#809)
    
    * initialize even if uniform is requested
    
    * update syrupy
    
    Former-commit-id: 39c2b694ae333fbcd15233384202132f772f0569

[33mcommit 4d197d95188d3699eb2aa6ed1d3637c233892641[m
Author: Misko <misko@meta.com>
Date:   Thu Aug 15 17:27:23 2024 -0700

    add example hydra config (#799)
    
    * add example hydra config
    
    * update ocp_hydra config
    
    * remove old hydra example
    
    * update ocp_hydra config
    
    Former-commit-id: 96f2835893a687fabcab7943c443eb9881648171

[33mcommit ef68eb7f9e54311cee1774f3e240b1468c338c8d[m
Author: Misko <misko@meta.com>
Date:   Thu Aug 15 17:24:02 2024 -0700

    load linref and normalizer on cuda device when resuming from checkpoint (#813)
    
    * load linref and normalizer on cuda device when resuming from checkpoint
    
    * move to() outside one level
    
    Former-commit-id: 19b2e24b85932d4b7daa2dbb1a6bb0986b661206

[33mcommit 99ea481e302150f29048dd62afd2250ddf2b362e[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Thu Aug 15 11:09:49 2024 -0700

    Activation checkpoint equiformersv2 (#811)
    
    * add back activation checkpoint
    
    * typo
    
    * add act check test
    
    * lint
    
    * rename test
    
    Former-commit-id: db6a042b10ad583d59d86967999866f4ea67386b

[33mcommit 58223d0f988ec07241134ebe24e7216df9c90ba8[m
Author: Misko <misko@meta.com>
Date:   Wed Aug 14 12:31:46 2024 -0700

    add resolution flag to escn (#804)
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 9c81cb53331a2dded469768fb83170f0d7ba9344

[33mcommit efb5dad966ea7c7b5f5ee4312a2d8afef2e58458[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Aug 13 16:09:51 2024 -0700

    Finetune Hydra (#797)
    
    * add basic function
    
    * support retain backbone mode
    
    * fix config
    
    * add hydra interface
    
    * run linter
    
    * run ruff
    
    * fix test
    
    * update main, add configs
    
    * add tests
    
    * test double finetune
    
    * format
    
    * fix few comments
    
    * remove finetuneinterface
    
    * update tests
    
    * remove configs
    
    Former-commit-id: 229c13ceedab0d8081899d971a80f4d62a439872

[33mcommit b1245437c53671848d82b4824eb36e4c74188695[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Aug 13 15:35:20 2024 -0700

    Fix job resume on non-preemption type failures (#803)
    
    * simplest way to fix resume on node failure
    
    * lint
    
    * add slurm util [y
    
    * add slurm.py
    
    * fix typo
    
    * add print
    
    * only modify pickle on master
    
    * fix comment
    
    * use submit to get pickle path
    
    * str paths
    
    * typo
    
    * typo
    
    * timestamp_id was already in cmd
    
    Former-commit-id: cc00e3a8c92b383bcc7676fcad0e1dfd953bdc23

[33mcommit bb6242b6a3935f6f74ef7fc0a2738b6bdaadff40[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Tue Aug 13 14:10:41 2024 -0700

    Release drafter config fix 2 (#808)
    
    * typo in filename for release-draft config for cattsunami
    
    * remove duplicate section in release drafts
    
    Former-commit-id: 27168e0718b468cbf0b7ffcd2b22605b75a9c159

[33mcommit 915eb434327fa58cea938d829b0ced87e24dee89[m
Author: Daniel Levine <levineds@meta.com>
Date:   Tue Aug 13 11:39:24 2024 -0700

    #806: Move data.db to s3 and download via download_large_files (#807)
    
    
    
    Former-commit-id: 75d8735c76f38cf4bcbe334a78d2f48ea7278f43

[33mcommit f24c0011c69f6d312f7306ebc198233eb6b8d5cd[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Tue Aug 13 09:37:24 2024 -0700

    typo in filename for release-draft config for cattsunami (#805)
    
    
    
    Former-commit-id: e3103a82d253cc7edd4a15a067bfabe63817081f

[33mcommit 74612d620fdb7267b08b2fdb624983e3aaff31c1[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Mon Aug 12 22:03:49 2024 -0700

    Automated semantic versioning and release drafter (#801)
    
    * test release drafter
    
    * master -> main
    
    * same categories for release-drafter as github release defaults
    
    * add v to version
    
    * version all components with the same calver and release all every time to imply they were tested together
    
    * no release draft on PR, only on push to main
    
    * ubuntu-latest
    
    * release drafter for each package
    
    * add dependencies as reasonable label
    
    * require reasonable label
    
    * rename checker
    
    * latest package only for core
    
    * small typos
    
    * small fixes to configs
    
    * remove contributors list as it already shows up in the release
    
    * small change to match original tags in repo
    
    * explicit read permissions for PRs
    
    Former-commit-id: 28448df669870b1d44c56ace21845e082ef90af9

[33mcommit 92061a9366b2aba0914e8520d71198c02f7cd38e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Aug 8 17:55:02 2024 -0700

    Add solvent interface placement code (#765)
    
    * Add solvent interface placement code
    
    * circular import
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * packmol install fix
    
    * add packmol to github actions path
    
    * speed up random slab generation
    
    * support for pymatgen update
    
    * save ion id when randomly sampled
    
    * vasp 6.3 ml flags are slightly different
    
    * add vdw to bulks
    
    * add lasph
    
    * ncore=4
    
    * typing, docstring, cleanup geometry
    
    * typing
    
    ---------
    
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Former-commit-id: 2bbc713ed077217c84fb9801dcb6848c47dcb18a

[33mcommit adba8aecec296637c18c3eea35d7dd76ec6d9de3[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Thu Aug 8 09:53:32 2024 -0700

    first commit (#798)
    
    
    
    Former-commit-id: 0055e4c9b50a1d883150c6108a241028db407b25

[33mcommit 877a693f6bb2bd677cbd350618cef99879043c97[m
Author: Misko <misko@meta.com>
Date:   Thu Aug 8 08:39:55 2024 -0700

    fix issues with ddp/hydra and add tests (#796)
    
    * fix issues with ddp/hydra and add tests
    
    * remove load balancing for painn tests
    
    * remove painn parameters when using hydra
    
    * update test configs to be consistent with each other
    
    Former-commit-id: 14798fd1da8f1dff46da94031f394815807291a8

[33mcommit d00d0f3ec7ae42059651e29e279e8e23e1a95de1[m
Author: Xiang <31351668+kyonofx@users.noreply.github.com>
Date:   Tue Aug 6 20:09:35 2024 -0700

    Wrap atom coordinates in ase.Atoms preprocessing (#783)
    
    * wrap atom coordinates in ase.Atoms preprocessing.
    
    * fix linting.
    
    * adjust atoms_to_graphs test cases.
    
    Former-commit-id: 42d66e7651a2f700c165dca7b9c7764d7e13f3b2

[33mcommit 05b9595a18bf99ab31f42a6a35298fc37c8b67b8[m
Author: Misko <misko@meta.com>
Date:   Mon Aug 5 19:02:52 2024 -0700

    Add an option to run PBC in single system mode (#795)
    
    * do pbc per system
    
    * add option to use single system pbc
    
    * remove comments
    
    * integrate use_pbc_single to all the models in repo; add test
    
    Former-commit-id: 90ae63a9178125436b50edb66d17213472488523

[33mcommit f2985b1375aa83dcd2ccdb046a5a90b61bcc9fd3[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Aug 5 17:05:03 2024 -0700

    update ocp example config (#794)
    
    
    
    Former-commit-id: ecee3390d175a0357f4c58866ff209937dec6480

[33mcommit ba1a791fa0636e2e64663d2c3d291c2717e007f4[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Sun Aug 4 21:14:35 2024 -0600

    (OTF) Normalization and element references (#715)
    
    * denorm targets in _forward only
    
    * linear reference class
    
    * atomref in normalizer
    
    * raise input error
    
    * clean up normalizer interface
    
    * add element refs
    
    * add element refs correctly
    
    * ruff
    
    * fix save_checkpoint
    
    * reference and dereference
    
    * 2xnorm linref trainer add
    
    * clean-up
    
    * otf linear reference fit
    
    * fix tensor device
    
    * otf element references and normalizers
    
    * use only present elements when fitting
    
    * lint
    
    * _forward norm and derefd values
    
    * fix list of paths in src
    
    * total mean and std
    
    * fitted flag to avoid refitting normalizers/references on rerun
    
    * allow passing lstsq driver
    
    * element ref unit tests
    
    * remove superfluous type
    
    * lint fix
    
    * allow setting batch_size explicitly
    
    * test applying element refs
    
    * normalizer tests
    
    * increase distributed timeout
    
    * save normalizers and linear refs in otf_fit
    
    * remove debug code
    
    * fix removing refs
    
    * swap otf_fit for fit, and save all normalizers in one file
    
    * log loading and saving normalizers
    
    * fit references and normalizer scripts
    
    * lint fixes
    
    * allow absent optim key in config
    
    * lin-ref description
    
    * read files based on extension
    
    * pass seed
    
    * rename dataset fixture
    
    * check if file is none
    
    * pass generator correctly
    
    * separate method for norms and refs
    
    * add normalizer code back
    
    * fix Generator construction
    
    * import order
    
    * log warnings if multiple inputs are passed
    
    * raise Error if duplicate references or norms are set
    
    * use len batch
    
    * assert element reference targets are scalar
    
    * fix name and rename method
    
    * load and save norms and refs using same logic
    
    * fix creating normalizer
    
    * remove print statements
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * warn instead of error when duplicate norm/ref target names
    
    * allow timeout to be read from config
    
    * test seed noseed ref fits
    
    * lotsa refactoring
    
    * lotsa fixing
    
    * more fixing...
    
    * num_workers zero to prevent mp issues
    
    * add otf norms smoke test and fixes
    
    * allow overriding normalization fit values
    
    * update tests
    
    * fix normalizer loading
    
    * use rmsd instead of only stdev
    
    * fix tests
    
    * correct rmsd calc and fix loading
    
    * clean up norm loading and log values
    
    * logg linear reference metrics
    
    * load element references state dict
    
    * fix loading and tests
    
    * fix imports in scripts
    
    * fix test?
    
    * fix test
    
    * use numpy as default to fit references
    
    * minor fixes
    
    * rm torch_tempdir fixture
    
    ---------
    
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 4b4b22f48348dca0bb1611c84b97ab7f4efe4335

[33mcommit eb85cd9f418eda43ffea7f9615d463ee5f6b6d9a[m
Author: Misko <misko@meta.com>
Date:   Fri Aug 2 13:50:24 2024 -0700

    Move select models to backbone + heads format and add support for hydra (#782)
    
    * convert escn to bb + heads
    
    * convert dimenet to bb + heads
    
    * gemnet_oc to backbone and heads
    
    * add additional parameter backbone config to heads
    
    * gemnet to bb and heads
    
    * pain to bb and heads
    
    * add eqv2 bb+heads; move to canonical naming
    
    * fix calculator loading by leaving original class in code
    
    * fix issues with calculator loading
    
    * lint fixes
    
    * move dimenet++ heads to one
    
    * add test for dimenet
    
    * add painn test
    
    * hydra and tests for gemnetH dppH painnH
    
    * add escnH and equiformerv2H
    
    * add gemnetdt gemnetdtH
    
    * add smoke test for schnet and scn
    
    * remove old examples
    
    * typo
    
    * fix gemnet with grad forces; add test for this
    
    * remove unused params; add backbone and head interface; add typing
    
    * remove unused second order output heads
    
    * remove OC20 suffix from equiformer
    
    * remove comment
    
    * rename and lint
    
    * fix dimenet test
    
    * fix tests
    
    * refactor generate graph
    
    * refactor generate graph
    
    * fix a messy cherry pick
    
    * final messy fix
    
    * graph data interface in eqv2
    
    * refactor
    
    * no bbconfigs
    
    * no more headconfigs in inits
    
    * rename hydra
    
    * fix eqV2
    
    * update test configs
    
    * final fixes
    
    * fix tutorial
    
    * rm comments
    
    * fix test
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: f47dd92e3ac2ac67ab88a29499ddd4b75f5e93f4

[33mcommit 93c676185706e0692ebf0b7e2f6bc9916d07c472[m
Author: Misko <misko@meta.com>
Date:   Fri Aug 2 10:19:10 2024 -0700

    Balanced batch sampler+base dataset (#753)
    
    * Update BalancedBatchSampler to use datasets' `data_sizes` method
    Replace BalancedBatchSampler's `force_balancing` and `throw_on_error` parameters with `on_error`
    
    * Remove python 3.10 syntax
    
    * Documentation
    
    * Added set_epoch method
    
    * Format
    
    * Changed "resolved dataset" message to be a debug log to reduce log spam
    
    * clean up batchsampler and tests
    
    * base dataset class
    
    * move lin_ref to base dataset
    
    * inherit basedataset for ase dataset
    
    * filter indices prop
    
    * added create_dataset fn
    
    * yaml load fix
    
    * create dataset function instead of filtering in base
    
    * remove filtered_indices
    
    * make create_dataset and LMDBDatabase importable from datasets
    
    * create_dataset cleanup
    
    * test create_dataset
    
    * use metadata.natoms directly and add it to subset
    
    * use self.indices to handle shard
    
    * rename _data_sizes
    
    * fix Subset of metadata
    
    * minor change to metadata, added full path option
    
    * import updates
    
    * implement get_metadata for datasets; add tests for max_atoms and balanced partitioning
    
    * a[:len(a)+1] does not throw error, change to check for this
    
    * off by one fix
    
    * fixing tests
    
    * plug create_dataset into trainer
    
    * remove datasetwithsizes; fix base dataset integration; replace close_db with __del__
    
    * lint
    
    * add/fix test;
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * Add extra test case for local batch size = 1
    
    * fix example
    
    * fix test case
    
    * reorg changes
    
    * remove metadata_has_sizes in favor of basedataset function metadata_hasattr
    
    * fix data_parallel typo
    
    * fix up some tests
    
    * rename get_metadata to sample_property_metadata
    
    * add slow get_metadata for ase; add tests for get_metadata (ase+lmdb); add test for make lmdb metadata sizes
    
    * add support for different backends and ddp in pytest
    
    * fix tests and balanced batch sampler
    
    * make default dataset lmdb
    
    * lint
    
    * fix tests
    
    * test with world_size=0 by default
    
    * fix tests
    
    * fix tests..
    
    * remove subsample from oc22 dataset
    
    * remove old datasets; add test for noddp
    
    * remove load balancing from docs
    
    * fix docs; add train_split_settings and test for this
    
    ---------
    
    Co-authored-by: Nima Shoghi <nimashoghi@gmail.com>
    Co-authored-by: Nima Shoghi <nimashoghi@fb.com>
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Brandon <bmwood@meta.com>
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: d0b679c2bbfab4f320de339fe0a0ec24e5c3237a

[33mcommit d0de201de8abe6ed1a5cc0f66f319c76b3e59c79[m
Author: Misko <misko@meta.com>
Date:   Thu Aug 1 21:31:08 2024 -0700

    list available fairchem packages; and show how to use pytest in [dev] (#790)
    
    
    
    Former-commit-id: 5bab41472fa3349800a227b375cf9d03d023e189

[33mcommit 1cc2411452db04e0564e6d5b44cde1616951fb7e[m
Author: Daniel Levine <levineds@meta.com>
Date:   Tue Jul 30 15:46:04 2024 -0700

    [BE] Remove large files from fairchem and add references to new location as needed (#761)
    
    * Remove large files from fairchem and add references to new location as needed
    
    * ruff differs from isort specification...
    
    * add fine-tuning supporting-info since it is over 2MB
    
    * add unittest
    
    * linting
    
    * typo
    
    * import
    
    * Use better function name and re-use fairchem_root function
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: dfd61132917d92c6c7f4a43a6d0ba349d728513a

[33mcommit 37c32e54adaa5eaf815b47c401fa2eb26e83104c[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Jul 30 10:34:22 2024 -0700

    clone so3 embedding object (#781)
    
    
    
    Former-commit-id: 1d8e70eeccf5509ff7221ba9606a7e0ca7dea0ae

[33mcommit 28ea78430b000d3936c5d3ea689ea791008b36d9[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Mon Jul 22 11:46:23 2024 -0700

    add expandable segments var (#775)
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * add expandable segments var
    
    * add note
    
    ---------
    
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 60ae7e5d9498a22c49d5a4a2e6246158a8b20fe8

[33mcommit edd53d59578f45505d7599d799c1d3b46573514a[m
Author: Haruyuki Oda <haruyuki.oda.ch@hitachi.com>
Date:   Sun Jul 21 07:00:51 2024 +0900

    add proxy and ssl support. (#778)
    
    * add proxy and ssl support.
    
    The "urllib3" module does not automatically load environment variables related to proxy and SSL settings, such as HTTP_PROXY, HTTPS_PROXY, and CURL_CA_BUNDLE.
    Some institutions's security environments need this change.
    The "requests" module loads these settings from environmental variables.
    
    * add requests module into project.toml
    
    Former-commit-id: c44665b4a22ef334e4a57ab343caedf5d76d1908

[33mcommit 768d5ac4fda0ede43a19211d27e6e8eede79b2ff[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Fri Jul 19 11:05:19 2024 -0700

    OCP->FAIRChem + paper highlight list in docs (#772)
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * OCP->FAIRChem + paper list
    
    ---------
    
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 5eb9d6b799c1942bdb00a431c30f7b6e0da7b11a

[33mcommit 8334e2c47add11f02ece12d651589cbf3fe9aa8e[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Jul 19 10:45:33 2024 -0700

    Fix dataset logic (#771)
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * fix dataset config logic
    
    * add empty val/test if not defined
    
    * add empty dicts for all missing datasets
    
    ---------
    
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
    Former-commit-id: d530571432a8f5bdcc3a1756259f21368126c35e

[33mcommit af9d893260fb5dcc9beedbaf25541ebc72d3c7b2[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Jul 19 06:35:02 2024 -0700

    Build docs test on submit only (#774)
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * trigger docs build only on PR review submit
    
    ---------
    
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 1367cd144fb702a8dd9d568a06e8cac791a27d12

[33mcommit dce72f18677ccc6a878c5eb2fbec116e489c8682[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Fri Jul 19 02:17:15 2024 -0400

    Fix NEB doc notebooks (#773)
    
    * adding new notebook for using fairchem models with NEBs without CatTSunami enumeration (#764)
    
    * adding new notebook for using fairchem models with NEBs
    
    * adding md tutorials
    
    * blocking code cells that arent needed or take too long
    
    * updating approach to path to work with ipython
    
    * adding seed to NRR example which randomly had not configs on last push
    
    * Update docs/tutorials/NRR/NRR_example.md
    
    * adding file :(|)
    
    * skip neb execution
    
    * status check on merge queue
    
    ---------
    
    Co-authored-by: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: f7f8b2b06cca695adca91d609fa0bf1e77f4c1b1

[33mcommit 1f07641acc3a86e71e617f9ebd6298db72aca0aa[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Sun Jul 14 11:56:04 2024 -0700

    fix docs badge, and add codespaces badge (#762)
    
    
    
    Former-commit-id: efac691fcf56489fe8446c8a5142679925ed984a

[33mcommit 20a8b406a3f01c58fb7845896b1a83366bf47c39[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Sat Jul 13 06:18:53 2024 +0300

    [BE] lint/test/docs on PR and push to main (#742)
    
    * lint/test on PR and push to main
    
    * Run docs CI on every push to main
    
    * Fail docs when a notebook doesn't build
    
    * skip eval_metrics fix in tutorial_utils
    
    * when writing extxyz in docs, specific force as a column explicitly
    
    * task.dataset -> dataset.format
    
    * small fixes in config in tutorial
    
    * make train dataset optional if only used test dataset
    
    * more small fixes if you don't specify a train dataset
    
    * loss_fns -> loss_functions in tutorial
    
    * eval_metrics -> evaluation_metrics in OCP tutorial
    
    * FixAtoms as int
    
    * don't clear train dataset if present but src is missing
    
    * invert train dataset load logic
    
    * fix logic
    
    * bool mask
    
    * require docs test on main
    
    * pull_request_review to trigger doc build
    
    * use FixAtoms fix from stress-relaxations branch
    
    * sneak in a push trigger to test docs
    
    * upload only html artifact
    
    * sneaky push
    
    * fix artifact path
    
    * fix job names
    
    * no deploy docs on push
    
    * add devcontainer settings
    
    * small fix to legacy tutorial
    
    * small fix to legacy tutorial
    
    * another small tutorial fix
    
    * add rest of tutorial back
    
    * clean up logic in load_datasets
    
    * typo
    
    * logic typo in load_datasets
    
    * convert all md to ipynb on launch
    
    * add minimum spec on devcontainer to make sure everything runs
    
    * include subdirectories in conversion
    
    * 758 fix
    
    * minor fixes
    
    * relax otf
    
    * typo
    
    * try removing comments in build_docs yml
    
    ---------
    
    Co-authored-by: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
    Co-authored-by: Zack Ulissi <zulissi@meta.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: da152c95058bdeca87deff82f4e1e33c1f8cddca

[33mcommit a918628aa5464ca5cb5f7a1bcc5e5c9f7224af04[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jul 12 14:32:17 2024 -0700

    Relax config (#758)
    
    * update example config per PR714
    
    * no need to support this
    
    * y->energy
    
    Former-commit-id: 67887fec1b4a7e725904a52823cd525ecb031d8c

[33mcommit b852c76d0f912c7f36e9cf818f04ba8efe5fbdb4[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Jul 12 20:20:36 2024 +0300

    [BE] Single ruff config (#751)
    
    * ruff toml
    
    * ruff fix
    
    * fix and format
    
    * update ruff
    
    * remove ruff config in pyproject
    
    * lint workflow use ruff.toml
    
    * unpin ruff
    
    * fix circular imports
    
    * clean up imports
    
    * revert to 3.9 annotations, dataclasses_json does not work with __future__ annotations
    
    * pin ruff
    
    Former-commit-id: 2353f41e2e388c6fc0541fa42b5f05e7927f3893

[33mcommit 8fc798110fbb9c65b82d812ba84a872d27f22aa3[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Thu Jul 11 10:39:11 2024 -0700

    Add slurm qos setting (#757)
    
    * add slurm qos setting
    
    * set defaults to none
    
    Former-commit-id: 871dc81cc2ba7b60af899168c220af1ce0861c08

[33mcommit 5296a193f8c6be8077dbed427f2bd08f15f739d3[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Jul 10 16:16:04 2024 -0700

    Make relaxation data more general (#714)
    
    * Changes to relaxation dataset
    
    * Changes to relaxation dataset
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: b477d18688bf11b527cdb26ea8e31fae3028b377

[33mcommit 767683c31d319cc1b8575c5c5b0295b190c512a5[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Jul 10 15:59:41 2024 -0700

    Add utils to help run torch profiling (#754)
    
    * add profiler to ocp trainer
    
    * add profiler utils
    
    * type check lint
    
    * fix typo
    
    * remove it from ocp_trainer
    
    * add comment
    
    Former-commit-id: dd195c6a101e0dd788f7d3028584de0dba5add8d

[33mcommit c966cb05c16e5938b09d7e6b6c84d9b78715aaa1[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Jul 9 16:06:48 2024 -0700

    Updated ODAC checkpoints & configs (#755)
    
    * Updated ODAC configs
    
    * Updated ODAC checkpoints
    
    Former-commit-id: 72553880a1489474b09822807e4ef1247da7c71e

[33mcommit 1d77e208a88e9bf015311371358be6bc03821cad[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Jul 2 15:03:04 2024 -0700

    Hide wandb.watch behind flag (#747)
    
    * hide watch behind flag
    
    * set frequency
    
    Former-commit-id: d96aa52df1267efbab698cac5927a355fc2849fc

[33mcommit 6af94663ec6d9c9cab5f5411e98ef78bf32b296e[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Tue Jul 2 13:37:04 2024 -0700

    Equiformers2 Graph Parallel (#740)
    
    * add graph parallel version of equiformer v2
    
    * fix tensor split sizes
    
    * Moved equiformers_gp code to inside equiformers code
    
    * log num params
    
    * typo
    
    * check logger exists first
    
    * update tb loggers
    
    * add debuger
    
    * add gp_utils tests
    
    * add eqv2 tests
    
    * lint
    
    * lint
    
    * typo
    
    * typo
    
    * gp -> use_gp
    
    * testing removing --import-mode importlib
    
    * bump pytest version
    
    * remove import-mode
    
    * log gp_gpus
    
    * change back to main pyproject.toml
    
    ---------
    
    Co-authored-by: Janice Lan <janlan@fb.com>
    Former-commit-id: a3eb22f3946106bf382b74d43e6218a5a5482a3e

[33mcommit 7a6907cc4eb84c324abe483b40c3603f6cfc5f28[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Mon Jul 1 08:56:16 2024 -0700

    try removing importlib (#746)
    
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: d9f60668dc34c4ffad0328d21719d1913a8ce770

[33mcommit 15487177437c5b3199205c0d19c8803f09a8b7b0[m
Author: Curtis Chong <curtischong5@gmail.com>
Date:   Fri Jun 28 16:56:26 2024 -0400

    move code to load datasets outside of the train dataset loader (#731)
    
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: a78162b2142f76ea96f7b080ca8736f8fe41da4f

[33mcommit bc2eed87c893ff76ec38c9d5573565bf1506f5e9[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Thu Jun 27 17:08:32 2024 -0700

    pin ruff to 0.4.10 (#745)
    
    
    
    Former-commit-id: 2a054f4933a21550f6421cdc0a31bebc840877d7

[33mcommit 2f3129f4543d1e6e1790540ade5c6d32612eb9ae[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Wed Jun 26 09:48:31 2024 -0700

    Log num params to W&B (#741)
    
    * log num params
    
    * typo
    
    * check logger exists first
    
    * update tb loggers
    
    Former-commit-id: 15efc068a56daab186e57aebd4ebbe2dbcfc84a7

[33mcommit db9a60b6a269a2f47ed234a15912c378cc062e3f[m
Author: Misko <misko@meta.com>
Date:   Tue Jun 25 14:44:46 2024 -0700

    Enable github-actions tests for all PRs even external forks (#738)
    
    * enable tests for all PRs even external forks
    
    * ignore push test on main
    
    * remove push add workflow_dispatch
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: b64811c2133ced12193b7c11954bb4c9e26655b5

[33mcommit 0daccce2e23ff879b9ee79aabcea7c789adea5ee[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Tue Jun 25 23:17:12 2024 +0300

    no ASE version pin (#739)
    
    
    
    Former-commit-id: 6254228fb312b1218b32dcc1d8b5436b31015c5b

[33mcommit 4acc872dd067cb5016b7473516cbcfe5b459f2cf[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Sat Jun 22 02:41:43 2024 +0300

    Config fix (#737)
    
    * config fix
    
    * config compatibility in ase calc
    
    * check if eval_metrics is present
    
    * handle existing empty loss_functions dicts
    
    Former-commit-id: a53a950b73ce03051cb0a1084c9740ad0f30230f

[33mcommit d2e5bf16bafe3da047937f341c8e6ccdbafa009f[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Fri Jun 21 15:38:57 2024 -0700

    Add option to launch distributed runs locally with >1 GPU (#733)
    
    * use torch elastic api to launch multiple gpu local mode
    
    * fix distutils.py
    
    * add test
    
    * lint
    
    * lint
    
    * basic test no distributed
    
    * update test
    
    ---------
    
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Former-commit-id: 8a984910fe5738eb11cdf7d3dcee26c76a3d5763

[33mcommit 82553270ce4004f99fa957888e4c839ca2353476[m
Author: rayg1234 <7001989+rayg1234@users.noreply.github.com>
Date:   Fri Jun 21 09:54:23 2024 -0700

    move ocpapi to integration tests (#736)
    
    * move to integration tests
    
    * remove code cov for integration
    
    * move integration to separate workflow
    
    * add change to see if integration test triggers
    
    * remove skip markers
    
    * change to on pull request
    
    * add test back, remove continue-on-error
    
    Former-commit-id: 9d6f1a56bad28c8a9435039a20b4198a834f66e4

[33mcommit 4e98ea681e111a61af24f19907b478b902a72e86[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Jun 20 13:59:33 2024 -0500

    calc compatability with new configs (#724)
    
    * calc compatability with new configs
    
    * make loss/eval naming consistent internally
    
    * numpy<2.0.0
    
    * cattsunami compatability
    
    * pin min numpy version
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Former-commit-id: 0d0a59ca731054c8465ff7ef30dad85849b40735

[33mcommit db6249e6c7a8d1cdb700309a7b9e4163c64a4715[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Thu Jun 13 10:32:45 2024 -0400

    Update adsorbml notebook (#728)
    
    * updating notebook to fairchem
    
    * remembering that we like to push notebooks that have been run
    
    Former-commit-id: eee29bdc43062d812bdc8bc88f49c6b94f7b314c

[33mcommit 0be2cd310367927e044f624213165fa946eac56e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jun 11 13:43:08 2024 -0700

    update preprocess script (#726)
    
    
    
    Former-commit-id: 693b9c6f93c37fd3305545a98ed3ae56cdf7f4cb

[33mcommit 2adb72f2d26c12a67eb23d08f7bab83e04502d3a[m
Author: Janice Lan <janlan@fb.com>
Date:   Tue Jun 4 17:31:04 2024 -0700

    [BE] Update all configs to use ocp2.0 format (#653)
    
    * switch all main repo configs to ocp2.0 format
    
    * move energy and force coefficients to loss functions
    
    * add force magnitude and remove extra energy metrics from s2ef configs
    
    * address PR comments
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 562c27be7957ad02561655598cc582a0a6b40b94

[33mcommit 3e740479b25ec64f1e658509b90f1fb26e4f137f[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Tue Jun 4 12:23:30 2024 -0700

    Bump ase from 3.22.1 to 3.23.0 (#720)
    
    ---
    updated-dependencies:
    - dependency-name: ase
      dependency-type: direct:production
      update-type: version-update:semver-minor
    ...
    
    Signed-off-by: dependabot[bot] <support@github.com>
    Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
    Former-commit-id: 2cc5e05a41334e03dd2e59c2b893e03d64f836f5

[33mcommit 413acbacfe3e0025e23209dd32fa0f799f96eac5[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Jun 3 20:22:04 2024 +0530

    K-points clarification in DAC Vasp settings (#716)
    
    * K-points clarification in DAC Vasp settings
    
    * Update setup_vasp.py
    
    Former-commit-id: 8097996f7ba77ef2c4b2f1b96d328cd18d5e1926

[33mcommit 85d29aece1e1a15b84d3d3d52c84c2ea9eee6d17[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu May 30 17:12:28 2024 -0700

    support logger entity (#717)
    
    * support entity
    
    * backwards support
    
    Former-commit-id: 4af4762cc24e1a660a9f8f4e69531756b3e5d149

[33mcommit 87804e0a880d3dcbc682aa9f4c6b518c0a9a56ae[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Tue May 28 12:25:17 2024 -0400

    fixing issue with one reaction (#712)
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 414179e1a6753a584c65453dd1f57efb49ad4655

[33mcommit ac7a8c0842c3c38133e306ff5cc12f7851046fb1[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed May 22 14:35:20 2024 -0700

    fix odac data links (#713)
    
    
    
    Former-commit-id: 5ec9b1824a5a4254a8755cf20e71df3ba4c065cf

[33mcommit b62516af51ac3513decafb0cdbc94cc7e288b94d[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed May 22 11:02:25 2024 -0700

    Added a file demonstrating ODAC Vasp settings (#711)
    
    
    
    Former-commit-id: 07dffb7dd81c7d97947aaee71f245592f22c8f27

[33mcommit e292ce8f2f4876375e248fe5cf3c612c06a1d30a[m
Author: Misko <misko@meta.com>
Date:   Wed May 22 09:33:04 2024 -0700

    pass seed to sampler (#697)
    
    * pass seed to sampler
    
    * fix up tests
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: a8e1a0699ba8bbda0d62f859731f3e0a7c2f3f78

[33mcommit b0509a38735af5b6d1e01af95e8f313ba032b6d4[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Wed May 22 08:26:30 2024 -0700

    --- (#707)
    
    updated-dependencies:
    - dependency-name: requests
      dependency-type: direct:production
    ...
    
    Signed-off-by: dependabot[bot] <support@github.com>
    Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
    Former-commit-id: fe1152f2bad629fe346a1fca2a65f55aead3b69b

[33mcommit 30eaf482d4a399bf55c748a1fa9a09f30d41bb29[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Wed May 22 11:09:41 2024 -0400

    adding seed to md tutorial and ipynb (#710)
    
    * adding seed to md tutorial and ipynb
    
    * removing tag to skip execution
    
    * Revert "removing tag to skip execution"
    
    This reverts commit 039f50b34e65ca885f6fa1f7204fb7ad4a0202ae [formerly 010365128c448070df563595b3d111d58a19fcaf].
    
    Former-commit-id: 3af0bfc86d3bcf3c847dd229cd093b519c34e615

[33mcommit b7fe1dea9be6f18d25bf70154b7809499292d7e3[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Tue May 21 14:13:56 2024 -0700

    doc exec auto (#706)
    
    
    
    Former-commit-id: bf26b017888625234cf6e72d67a53b42ed35fb15

[33mcommit 74bab0b4b01ac447bdc2fddf89bcbf04f37af8e6[m
Author: Misko <misko@meta.com>
Date:   Fri May 17 10:36:20 2024 -0700

    update install instructions (#705)
    
    * update install instructions
    
    * fix up install instructions
    
    * documentation build breaking, lets try auto
    
    * double notebook timeout
    
    * dont execute cattsunami tutorial in docs
    
    * no logo for now
    
    * final edits
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Former-commit-id: cae5af2c560bcae305dae306321e78c7810d62f1

[33mcommit d1d14d224673209ae70656e0a4ef0931194ab4d1[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Thu May 16 09:41:15 2024 -0700

    fixes (#704)
    
    
    
    Former-commit-id: 1f7545620e8e3df2ff36d8298c6ec7929543f157

[33mcommit 196de0fbd79bf5a16f81ca91508bfd514c489b97[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed May 15 18:01:29 2024 -0700

    cleanup amp warning (#703)
    
    * cleanup amp warning
    
    * rm redundant logic
    
    * add version in base_trainer and supress stderr when getting commit_hash
    
    * ruff
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Former-commit-id: 3cb004df3f7cdf47ae237512a67e6763b6c13f07

[33mcommit 9660d1e326247054548d0294c42d23a88e9333f9[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Wed May 15 17:54:00 2024 -0700

    update install instructions (#702)
    
    * update install instructions
    
    * minor edit
    
    Former-commit-id: a87061d9f72ab1294a4276469188c30119b0a27a

[33mcommit 1387a4fd8827e25c8d64747c7744e5962b7ca134[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Wed May 15 17:49:38 2024 -0700

    Package cleanup (#698)
    
    * add cattsunami package dependencies
    
    * add data-oc package dependencies
    
    * update install and dependencies
    
    * update dependencies
    
    * typo
    
    * fix parentheses in release statement
    
    * release inputs only on dispatch
    
    * remove superfluous check of release.tag_name
    
    * add urls to packages
    
    Former-commit-id: 0c87e013142f909334cf5c6eb3ef340bb6dafa86

[33mcommit 348df1f94a4c3485c6f1aeb22346026d00690e04[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Wed May 15 18:57:58 2024 -0500

    fixing summary fig and dissociation scheme fig in the gitbook (#700)
    
    * fixing summary fig and dissociation scheme fig in the gitbook
    
    * adding pip cattsunami to readme
    
    Former-commit-id: b4feee9fb509bb2e95cac03bd2a3544c8276b6ae

[33mcommit f810caa827fb0a8b0ba1403d851a6b28536c99b9[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed May 15 16:01:05 2024 -0700

    reorganize toc (#701)
    
    
    
    Former-commit-id: 645f7e5bee6eea83513526334bf3ca6825276660

[33mcommit 4f761c8312c38901a6b821b555b076ac1a844056[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Wed May 15 16:59:14 2024 -0500

    adding pages for oc20dense and oc20neb (#699)
    
    * adding pages for oc20dense and oc20neb
    
    * minor edit
    
    * minor edits
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 6d3e84eeead0993e8972154623b388e6ebbe242f

[33mcommit f009640c7fddc3f8464f1760677f9fce1c59f93b[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed May 15 10:56:11 2024 -0700

    simplify calc usage (#694)
    
    * simplify calc usage
    
    * Add quick start
    
    * trying to restore model_checkpoints.md
    
    * update precommit version
    
    * precommit
    
    * add error when using both model_name and checkpoint_path
    
    ---------
    
    Co-authored-by: Misko <misko@meta.com>
    Former-commit-id: d71e9acd7e049eb5842939c836b4e983ebf4495e

[33mcommit 9ee58907f9abf9b38f56b21ce806335c46dcef08[m
Author: Misko <misko@meta.com>
Date:   Wed May 15 10:55:58 2024 -0700

    skip ocpapi integration tests in default test workflow (#696)
    
    * skip ocpapi integration tests
    
    * missed conftest include
    
    Former-commit-id: 1f7b1a4f4c3f101cb9d31a7d288e8233717117e2

[33mcommit 14248d69cda56f0c6ae9c0b2106865bf1f494ccf[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Wed May 15 05:39:19 2024 -0700

    fix dist paths in release workflow (#695)
    
    
    
    Former-commit-id: 4c1d2b6c37a7c4be2de1715f3eecf434f5d5cf8a

[33mcommit 245d62ac57914ec22f352404113b85a556f0c162[m
Author: Misko <misko@meta.com>
Date:   Tue May 14 21:43:21 2024 -0700

    add windows install instructions (#692)
    
    
    
    Former-commit-id: c13fcedcea9beb0203d50288b14c6f45afed3bb7

[33mcommit 12ef5b7ac2207bb6f5dc75df0752202ea55ac6bb[m
Author: Misko <misko@meta.com>
Date:   Tue May 14 21:43:08 2024 -0700

    change cattsunami autoapi and skip ocpapi execution from docs (#693)
    
    * add fix for autoapi
    
    * skip execution of cells with ocpapi tests because of rate limiting
    
    Former-commit-id: 2ffbf4c28066d112322a19004bcaed5856843a0e

[33mcommit d04f1bb2eec480acf6804f94cfe73ddbb695bb0b[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Tue May 14 15:09:41 2024 -0700

    PyPi CI workflows (#690)
    
    * build distributions separately
    
    * draft release workflows
    
    * build V5
    
    * test-pypi to test
    
    * no docs for right now
    
    * check ref_name
    
    * no test for now
    
    * set test environment for testing
    
    * environments
    
    * add more release packages
    
    * fix dist paths
    
    * final release and test-release workflows
    
    * add cattsunami and fix project urls
    
    * new fairchem-core readme
    
    * use test-pypi in environment url
    
    * update ocpapi README.md
    
    * fix only-include
    
    * add version
    
    * add version
    
    * no trigger on release for test-release.yml
    
    * temporary test-release as release
    
    * revert back to real release
    
    * add cattsunami to build.yml
    
    * revert real release.yml
    
    Former-commit-id: 9db4163a0691feb90d95eab3c63721c419e4f8eb

[33mcommit bab6d85279c4e517802975e1d7322115c96ba038[m
Author: Misko <misko@meta.com>
Date:   Tue May 14 13:31:35 2024 -0700

    Small fixes to make tutorial notebooks pass (#688)
    
    * fix update_config
    
    * fix update_config
    
    * rename eval_metrics
    
    * wget links and lint
    
    * update gemnet model name
    
    * add TODO in code
    
    * lint
    
    Former-commit-id: 9b92d59848f7e98ace37faef33251ed4fdc99ff1

[33mcommit c6e7ba587ab492ea45c9b8032297874684ef1d54[m
Author: Misko <misko@meta.com>
Date:   Tue May 14 09:57:56 2024 -0700

    [BE] Add smoke test for [escn,gemnet,equiformer_v2] train+predict, Add optimization test for [escn,gemnet,equiformer_v2] (#640)
    
    * Add a simple pickle dataset type and a test case for escn training
    
    * fix import
    
    * lint
    
    * wrong paths
    
    * circleci gets a bit diff result than local, add buffer
    
    * Add S2EF e2e test
    
    * working e2e smoke test and short optimizer tests
    
    * remove unused pickle dataset support and data files
    
    * add torch deterministic
    
    * lint
    
    * lint again
    
    * clean up tests using parameterize, add tests for predict
    
    * lint
    
    * remove unused imports from test_escn
    
    * fixes
    
    * lint
    
    * fix lint
    
    * fix yaml paths
    
    * correct scaling path
    
    * promote up tests folder
    
    * fix up tests
    
    ---------
    
    Co-authored-by: Richard Barnes <rbarnes@umn.edu>
    Former-commit-id: fae193c8823cb65eaefad2c9651e28d3f06bf655

[33mcommit f48275c597b4e74cbfc1b98061b0a53454519dd8[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon May 13 21:13:46 2024 -0700

    cattsunami packaging (#678)
    
    * pyproject.toml for cattsunami
    
    * adding tests for cattsunami
    
    * ocpneb packaging
    
    * cleanup ocpneb tests
    
    * normalize project names
    
    * move cattsunami tests to root test folder
    
    * fix imports
    
    * allow ase imports from latest version and development version
    
    * pip install cattsunami in tests
    
    * cleaning up paths and getting tests to function
    
    * install cattsunami for docs as well
    
    * sneak-in a doc syntax fix
    
    * no space
    
    * updating doc md
    
    * no pytest in reaction
    
    * refactor ocpneb package
    
    * fix ocpneb imports
    
    * move README.md
    
    * fix ocpneb install
    
    * rename test directory
    
    * test lint
    
    * renaming to cattsunami
    
    * updating gitbook build with package name
    
    * changing runner test yaml to cattsunami
    
    * changing adsorbates pkl to adsorbate pkl
    
    * update cattsunami dist build directory
    
    * updating readme with monorepo links
    
    * trimming down test time
    
    * reducing batch size incase of mem issues
    
    ---------
    
    Co-authored-by: Brook Wander <brook.l.wander@gmail.com>
    Former-commit-id: d3058d79d85bebfe06f8a6f9cc39cce860e6dad7

[33mcommit 333a6baac9595013174d71f500643bcb6869c60e[m
Author: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
Date:   Mon May 13 15:52:21 2024 -0700

    Bump actions/setup-python from 4 to 5 (#689)
    
    Bumps [actions/setup-python](https://github.com/actions/setup-python) from 4 to 5.
    - [Release notes](https://github.com/actions/setup-python/releases)
    - [Commits](https://github.com/actions/setup-python/compare/v4...v5)
    
    ---
    updated-dependencies:
    - dependency-name: actions/setup-python
      dependency-type: direct:production
      update-type: version-update:semver-major
    ...
    
    Signed-off-by: dependabot[bot] <support@github.com>
    Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
    Former-commit-id: cc70eaaa9b754d524d0b910b5706671f0e9b8ad9

[33mcommit 2ca6fd203005b3b2e9a3749757df269d991edc0b[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Sat May 11 02:22:08 2024 +0200

    Fixes and updates to the documentation and website (#676)
    
    * first docs fix commit
    
    * fix repo url
    
    * fix bad relative imports
    
    * fix api docs using implicit namespaces
    
    * misc link fixes in docs
    
    * fix model names
    
    * remove papers_using_models from second mention
    
    * small fixes to api links
    
    * more link fixes
    
    * more link fixes
    
    * try to deploy to external repo
    
    * try to deploy to external repo
    
    * force execute notebooks
    
    * small fix for ruff, and fix docs dir
    
    * push to gh-pages
    
    * add inits, change gitignore
    
    * continue to rename to fairchem in docs; feel free to roll this back
    
    * misc docs fixes
    
    * more small fixes
    
    * fix main.py location in tutorial utils
    
    * gotchas checkpoint fix
    
    * fairchem_root
    
    * fix build by removing __init__.py
    
    * add env to conda command
    
    * edit install docs
    
    * update sphinx targets
    
    * execute auto
    
    * execute force
    
    * publish for the moment to gh-pages branch aswell
    
    * quite install ocpapi
    
    * fix finetuning from notebook tutorial
    
    * try to remove scale file from tutorial
    
    * download scaling file in tutorial
    
    * edit conf.py
    
    * fix some links
    
    * wget al scale files
    
    * move configs to root
    
    * temp links to build docs
    
    * update index page
    
    * update docs logo
    
    * set scale file links to main branch
    
    * set scale file links to main branch blob
    
    ---------
    
    Co-authored-by: Misko <misko@meta.com>
    Co-authored-by: lbluque <lbluque@meta.com>
    Former-commit-id: 5dc6bcbc53704e404f214da0afde1ecc83e06f08

[33mcommit d9d3962f8f324fa8aa241b3602f1c23065929e08[m
Author: Misko <misko@meta.com>
Date:   Fri May 10 16:57:46 2024 -0700

    Consolidate tests into a single folder (#682)
    
    * add init to fix pytest
    
    * add init to fix pytest
    
    * fix ga workflow for test
    
    * add pyproject toml config to pytest
    
    * remove collect only from pytest
    
    Former-commit-id: d4d5a6c87d3fca920a3f74e7a9ed65b947be1318

[33mcommit c0f794c0f222f797a95e73ac245e646a0d39a196[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri May 10 16:53:26 2024 -0700

    logo colors (#687)
    
    
    
    Former-commit-id: 57ad5287e120bfba0632e1bb0718f5fa70bfa072

[33mcommit 4b88318afbf48425ab7fe65b1a1274907dc10e78[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri May 10 15:23:39 2024 -0700

    Update main readme (#686)
    
    * initial README update
    
    * Update README.md
    
    * Update README.md
    
    * Update README.md
    
    * badge goodies and center headers
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Former-commit-id: 60d0db8e9a226cb2186590e802a8f4bd2fbdb6cb

[33mcommit 980cc8798c359e4c1e9d7548ba08cf31843b7128[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri May 10 08:21:42 2024 -0700

    GA workflows update (#675)
    
    * install from root
    
    * install from root and build docs on call or dispatch
    
    * no pushd in tests
    
    * test core at end to have that coverage reported for now
    
    * build packages
    
    * setup python
    
    * upload everything
    
    * fix dist building
    
    * fix build dists
    
    * add README.md
    
    * update dynamic vcs
    
    * update package urls
    
    * normalize project names
    
    Former-commit-id: c2a53285182bc724efe2c46564bd29abd235a609

[33mcommit a4cb8645b49539c5212e18fe1ce1c1abc6573ade[m
Author: Misko <misko@meta.com>
Date:   Thu May 9 19:08:45 2024 -0700

    make conda cpu env name consistent with gpu version (#683)
    
    
    
    Former-commit-id: bbcb6a2ea3b54c0ba88d1117508feee7fd11f76f

[33mcommit 5597a909b7bed2d8c804fc6f2afd3cbd2fe9e484[m
Author: Misko <misko@meta.com>
Date:   Thu May 9 14:51:03 2024 -0700

    fix ocpapi test (#681)
    
    
    
    Former-commit-id: 587a731d60de9e595a3ed40543b381be04e53785

[33mcommit c2dcd0e039b95ee8b7493940a6aeac8f4d6f600b[m
Author: Misko <misko@meta.com>
Date:   Thu May 9 11:03:19 2024 -0700

    fix ocp readme links (#679)
    
    * fix ocp readme links
    
    * fix ocp readme links
    
    * fix ocp readme links
    
    Former-commit-id: cc902b289358c9d5a4578329e407096cbb1030d8

[33mcommit adc7104e20caf2811cf3d546024727d507654991[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Wed May 8 17:51:08 2024 -0700

    fix install order so pyg does not mess things up (#673)
    
    
    
    Former-commit-id: c787ffb16fe8401657b600b4143e2d7bb5c314c8

[33mcommit 7665d202c1192993bdccf71902b954caae0c1411[m
Author: Misko <misko@meta.com>
Date:   Tue May 7 14:51:21 2024 -0700

    Monorepo (#670)
    
    * Update README.md
    
    Change the recommended threshold for anomaly detection.
    
    * Update flag_anomaly.py
    
    Update surface atom movement threshold for surface reconstruction detection.
    
    * changes to make OC-Dataset compatiable with newer version of ASE and pymatgen
    
    * move test folder into ocdata for easier import of the helper functions (#46)
    
    * implemented changes
    
    * updating with reverse method applied too
    
    * lint
    
    * making requested updates
    
    * Initial commit
    
    * Update and rename LICENSE to LICENSE.md
    
    * Update README.md
    
    * Update README.md
    
    * add model configs
    
    * pre-commit checks
    
    * Upload OC20-Dense dataset links
    
    * Add metadata information
    
    * readme typo
    
    * Update mapping checksum
    
    * adsorbml eval script
    
    * update docstring
    
    * Update README.md
    
    * add scripts and readme initial version
    
    * add utility script
    
    * some documentation
    
    * Update README.md
    
    * Update README.md
    
    * Create README.md
    
    * Update README.md
    
    * Update README.md
    
    * Update README.md
    
    * pre-commit hooks
    
    * fix eval ionic speedup
    
    * update db/pkls to new ASE
    
    * random placement code
    
    * update READMEs
    
    * rename var for consistency
    
    * add pointer to pretrained checkpoints
    
    * Update MODELS.md
    
    * add ml+sp success rates
    
    * Bugfix: sample at least 1 site when no. of simplices > num_sites
    
    * Bufix: pymatgen expects rotation angle in radians not degrees
    
    * Shuffle sites before returning num_sites
    
    * ceil instead of floor
    
    * set center of mass to 0,0,0 before placing adsorbate
    
    * add setup file
    
    * reorg for setup
    
    * set binding adsorbate to site
    
    * update com docs
    
    * inplace bufix
    
    * eval backwards compatibility fix
    
    * adding notebook
    
    * updating some small stuff
    
    * fix min_diff abs ordering; fmax on only moveable atoms; minor readme fixes
    
    * Placement code refactor (#62)
    
    * Reorganizes db files a little bit
    
    * flake8 config
    
    * Reorganize + lint old tests
    
    * Separating out core classes + runner code
    
    * default python gitignore
    
    * Runner scripts in a separate folder
    
    * Bulk refactor
    
    * Renames surfaces to surface
    
    * Default paths
    
    * Specify types in core.Bulk
    
    * core.Adsorbate refactor + init tests
    
    * Surface --> slab in core.Bulk
    
    * core.Surface refactor
    
    * Copy over atoms when initializing bulk / adsorbate
    
    * Some helper functions for adsorbate, bulk, surface
    
    * Adslab generation, including rigid body rotations along x, y, z
    
    * Structure --> core.Adslab
    
    * More renaming
    
    * Docs
    
    * vasp_flags change from main
    
    * remove comment
    
    * adding my changes since they are pretty extensive. Needs testing still.
    
    * operational. Known issue: itersecting with atoms outside of the simplex - easy fix anticipated
    
    * making proximate placement optional since it has the small issue
    
    * commit for posterity, changing approach
    
    * new approach works well, but definitely clunky. Still needs testing.
    
    * init ci config
    
    * install typo
    
    * pass black
    
    * cache ci build
    
    * add ci badge
    
    * update tests, paths, bulk db
    
    * relative ci path
    
    * ci directory structure fix
    
    * set precomputed defaults to None
    
    * update adslab test, add vasp test
    
    * downgrade codecov
    
    * rename db folder, import error
    
    * Rename `Surface`-->`Slab`, `Adslab`-->`AdsorbateSlabConfig`
    
    * Moves a lot of the slab creation logic to the Slab object
    
    * Pull symbols and covalent radii from ase instead of hardcoding it here
    
    * Makes surface tagging modular
    
    * Initializing a slab from ase.Atoms
    
    * Minor update to how Slabs are initialized from atoms directly
    
    * Return adsorbate smiles if available
    
    * Pmg placement heuristics: on-top, bridge, hollow sites + binding atoms
    
    * Updates test_inputs to latest api
    
    * Pass tests
    
    * update conversion script
    
    * Removes hardcoded min_xy
    
    * Constrains heuristic rotations around x and y
    
    * bugfix: bulk_id_from_db wasn't getting set when randomly sampling a bulk
    
    * explicitly solving for the point of intersection
    
    * cleaned up docs
    
    * making it work with heuristic and fixing par propagation
    
    * Fix undersampling around the edges
    
    * Update bulks pkl
    
    * Drop unused constants
    
    * Updates docstring
    
    * Fix seed to set kpoints deterministically
    
    * Remove unused functions from vasp utils
    
    * AdsorbateSlabConfig doc update
    
    * all functional - will clean up docs and what args are passed
    
    * all cleaned up. some bad placements possible. need to investigate
    
    * added runner script (docstrings not finalized yet)
    
    * correctly center at site
    
    * bugs with placement
    
    * generate from file of indices
    
    * correcting adsorbate position for projection
    
    * cleaning up overlap function
    
    * adding tests + cleanup
    
    * return missing
    
    * pool for runner script, and formatting
    
    * add codecov badge
    
    * uncommenting because this edge case still exists
    
    * save precomputed surfaces if doesn't exist
    
    * adding intercalation test and moving anomaly detection
    
    * formatting :)
    
    * formatting :) :)
    
    * save tiled and tagged slabs, only tile one for random if not saving
    
    * tqdm support for driver script
    
    * New mode: random sites + heuristic adsorbate placement
    
    * making bidentate a random choice
    
    * Adsorbate is already moved to site; avoid extra param
    
    * add precompute slab logic
    
    * add missing pool
    
    * save out site and sampled rotation angles to metadata
    
    * Uniformly random 3d rotations
    
    * bug fix - now check for atoms withing interstitial gap to solve for intersections
    
    * make bulk db searchable by src id
    
    * adding adsorbate searchability via SMILES
    
    * adding bulk docs
    
    * removing check for 2D materials since we plan to just catch errors at anomaly detection
    
    * adding ability to enumerate specific miller indices
    
    * set default and flags for random placement rotations
    
    * write surface inputs as part of precompute step
    
    * rand site+heur diff prefix
    
    * delete file only if it exists
    
    * adding binding index selection to adosrbate class for the case where atoms are provided as input.
    
    * Updating readme to reflect current repo state
    
    * updates to readme api
    
    * remove outdated precomputed surfaces from readme
    
    * Make things pip-installable
    
    * Readme pass
    
    * Minor updates to readme
    
    * Pass on ocdata.core docs
    
    * Link to the OC20 commit
    
    * Minor
    
    * Version bump
    
    ---------
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: Brook Wander <brook.l.wander@gmail.com>
    Co-authored-by: Janice Lan <janlan@fb.com>
    
    * Fix repo badges (#64)
    
    * remove old dataset logic
    
    * updates imports
    
    * rm unnecessary lines
    
    * requested changes from Muhammed
    
    * log successful completion
    
    * adding output
    
    * removing paths
    
    * add ml-only success metric
    
    * rename to free atoms
    
    * initial changes - think this looks much faster
    
    * cleaning up timing
    
    * black on files I havent changed
    
    * add no vasp flag to precompute slabs
    
    * Fix NH2NH2 naming and add reaction strings to adsorbates pkl (#67)
    
    * fix NH2NH2 naming
    
    * rm ase dbs, add reaction string
    
    * rm conversion script
    
    * resolve failing tests
    
    * Allow pre-loaded databases in initializers
    
    Bulk and Adsorbate initializers accept a path to database stored in a
    pkl file. This adds an option to instead pass in an already-loaded
    database to support cases where they are being read outside of these
    classes.
    
    * Fix test for new adsorbate db structure
    
    * Backwards compatibility for adsorbate db (#70)
    
    * Backwards compatibility for adsorbate db
    
    https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset/pull/67
    added support for reaction strings in adsorbate database entries. Tuple
    unpacking in the Adsorbate initializer assumes length 4 (after the
    reaction string was added). Older databases will be missing the reaction
    string and the initializer will raise when they are used. This makes the
    initializer backwards compatible with older databases, saving the
    reaction string only if present.
    
    * _save_adsorbate -> _load_adsorbate
    
    * adding 2023 neurips challenge folder with eval script and mapping/target files
    
    * adding a readme for challenge_eval script
    
    * Initial commit
    
    * Add Client and get_bulks method
    
    * Use common test cases for client method
    
    There are a few test cases that we'll want to run against each client
    method: unexpected response code, exception raised in client, and
    successful response. This adds a function that runs all cases for each
    client method, rather than copying the code around for each test.
    
    * Add support for GET /ocp/adsorbates
    
    * Remove try/catch when reading api response
    
    This was left over from parsing the response with an assumption that is
    was json serialized. Now we are just fetching the plain text in the
    response.
    
    * Removed defaults in data models
    
    After getting through a few more data structures in the API, there are
    enough fields without reasonable defaults that I think it makes sense to
    removed defaults. We'll still handle the addition of new fields, but
    removal of a field, if that happens at some point, will require users to
    upgrade their library version.
    
    * Add support for slab enumeration
    
    * Add support for enumerating adsorbate slab configs
    
    * Add support for submitting relaxations
    
    * Add support for fetching relaxation results
    
    * Add support for deleting relaxations
    
    * Simplify model naming
    
    * Add support for fetching relaxation requests
    
    * Add doc for exceptions in client methods
    
    * Add more type hints to client
    
    * Add exception type for rate limited api calls
    
    * Add exception type for calls that cannot be retried
    
    * Add public api to __init__ file
    
    * Add retry decorator for calls to the ocp api
    
    * Create client module
    
    * Add workflows for adsorbate slab relaxations
    
    * Add context to log statements
    
    * Add logging for rate limited requests
    
    * Return adsorbate details from workflow
    
    * Add quickstart to readme
    
    * Rename filter on miller indices
    
    * Add file IO example to README
    
    * Rename AdsorbateConfiguration to AdsorbateSlabRelaxation
    
    * Add default client to public workflow methods
    
    * Add lifetime to find_adsorbate_binding_sites
    
    * Add integration tests for workflows
    
    * Add unit tests for context.py
    
    * Add unit tests for retry.py
    
    * Add "slab" to public workflow names
    
    * Add unit tests for adsorbates.py
    
    * Add support for omitted_config_ids fields
    
    * Add progress bar that tracks finished relaxations
    
    * Add note about asyncio to readme
    
    * Add methods to convert to ase.Atoms objects
    
    * Add forces/energies to ase.Atoms when possible
    
    * Add FixAtom constraint to sub-surface atoms
    
    * Do not flatten relaxation results across slabs
    
    * Configure setuptools to make the package pip-installable  (#11)
    
    * Initial setup.py
    
    * Readme note
    
    * Remove env yaml
    
    * Make sure to run relaxations with tagged slabs
    
    * Create client with scheme and host
    
    Previously the Client initializer took the full base URL as input. This
    splits into scheme and host inputs. Later we'll use the host to map API
    urls to UI urls.
    
    * Add method to get UI results URL
    
    Adds a method that returns the URL at which results can be visualized.
    
    * Add URLs to workflow outputs
    
    Adds the API host and UI URL for each set of relaxations (system in API
    terminology).
    
    * Use equiformer v2 as the default model type
    
    * Pypi distribution (#15)
    
    * pypi install update + add license, citing info to setup, readme
    
    * Add patch id to version
    
    * Implement remaining unit tests
    
    * Add circleci config
    
    * Add circlci badge and code coverage
    
    * Check allowed models against server side list
    
    * More general slab filtering
    
    The old slab_filter interface accepted a single slab and returned
    True/False to keep/reject it. This updates the interface to accept a
    list of adslabs and return a list of adslabs, allowing for operations on
    the entire set, including on individual adsorbate configs if needed.
    
    * Add prompt_for_slabs_to_keep filter
    
    Changes the default behavior of find_adsorbate_binding_sites() to prompt
    users for the set of slabs that they want to submit.
    
    * Update docs about "adslab"
    
    * Documentation fixes
    
    Adds a note to the README about the supported bulks and adsorbates,
    fixes some language in documentation throughout the package, and
    modifies some docstrings so that they work well with sphinx
    documentation generators.
    
    * Release v1.0.0
    
    * Initial commit
    
    * Added force field evaluation code (#1)
    
    * Added force field evaluation code
    
    ---------
    
    Co-authored-by: Anuroop Sriram <anuroops@meta.com>
    
    * Add files via upload
    
    * skip vasp surface inputs + Pymatgen install fix (#72)
    
    * skip vasp surface inputs
    
    * fix logic
    
    * fix ci
    
    * ci debug:
    
    * snapshot pymatgen version
    
    * ci debug
    
    * pmg conda install
    
    * pmg install ci debug
    
    * update pmg debug
    
    * pass tests
    
    * missing tests
    
    * test fix
    
    * update shift
    
    * Add files via upload
    
    * Add files via upload
    
    * Add files via upload
    
    * Support for placing multiple adsorbates (#74)
    
    * initial multi-adsorbate placement
    
    * support for multi configurations
    
    * add multi-ads coverage tests
    
    * update docstring
    
    * typo
    
    * update metadata dict
    
    * adsorbate argument to placement
    
    * Remove unused imports
    
    * Remove more unused imports
    
    * update docs
    
    * indent
    
    * Update README.md
    
    * Update README.md
    
    * init commit
    
    * pre-commit hooks
    
    * add license
    
    * add gitignore
    
    * Add initial Architector examples for testing (#1)
    
    * Add initial Architector examples for testing
    
    * rm checkpoint
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    
    * add MIT license (#77)
    
    * add MIT license
    
    * add license to readme
    
    * add orca recipes (#3)
    
    * add orca recipes
    
    * more docs
    
    * explicit setup, directory
    
    * version fix
    
    * support quacc==0.7.2
    
    * include pkls in pip setup (#81)
    
    * [BE] Remove hardcoded paths (#82)
    
    * include pkls in pip setup
    
    * remove hardcoded paths
    
    * remove unused configs
    
    * fix test imports
    
    * be moving too fast
    
    * use pyproject.toml
    
    * github actions
    
    * in workflows directory
    
    * in workflows directory
    
    * correct call to black
    
    * check only 3.9
    
    * fix dependencies
    
    * ocdata not ocpdata
    
    * Store oriented unit bulk (#75)
    
    * save out oriented unit bulk
    
    * informational codecov
    
    * black diff
    
    * fix lint
    
    * test 3.9 only...
    
    * initial commit
    
    * moving stuff around and correcting paths
    
    * bare bones for validation -- files need updating
    
    * 3.9 - 3.11
    
    * general updates
    
    * Add additional Orca keywors for population analysis/properties which add minimal cost (#4)
    
    * Add additional Orca keywors for population analysis/properties which add minimal cost
    
    * actually NormalPrint doesn't add useful things
    
    * update Sella optimizer to fmax 0.05
    
    ---------
    
    Co-authored-by: Daniel Levine <levineds@meta.com>
    
    * remove Field
    
    * all cleaned up -- should be ready for review
    
    * clearing cell output from tutorial notebook
    
    * clean up timing
    
    * reducing number of initial configs
    
    * updating file path approach to mirror new approach in ocdata
    
    * explicitly define sella kwargs (#7)
    
    * explicitly define sella kwargs
    
    * lower scf+max_steps
    
    * Support for custom Orca calculator (#6)
    
    * feje orca support
    
    * feje support
    
    * return results
    
    * updates per latest quacc
    
    * add nbo
    
    * move monkey patch to driver script, not recipes
    
    * pin ase+update quacc
    
    * clarify docs
    
    * Added supercell info file
    
    * Update README.md
    
    * scripts for sampling GEOM dataset (#5)
    
    * scripts for sampling geom
    
    * Update biomolecules/geom/sample_geom_drugs.py
    
    Co-authored-by: Daniel Levine <levineds@gmail.com>
    
    * minor updates based on review
    
    * updating geom scripts to new file structure
    
    * fixed level of sampling
    
    ---------
    
    Co-authored-by: Daniel Levine <levineds@gmail.com>
    
    * add sella
    
    * modify nbo input (#9)
    
    * Add printing of Reduced Mulliken and Lowdin populations for each orbital (#10)
    
    Also added Lowdin and Mulliken bond orders since they were the only population feature missing from the NormalPrint level.
    
    * mark slabs test xfail per pmg bug
    
    * assert approx
    
    * no pmg version, lint, remove circleci
    
    * grid3
    
    * Update README.md
    
    fixing tutorial link and adding dataset download link
    
    * adding arXiv links to readme
    
    * add checksum
    
    * move to new src folder
    
    * rename imports for monorepo
    
    * fix up .gitignore @lbluque solved .pt include
    
    * folder promote ocpapi and open-catalyst-dataset
    
    * fixes to workflow
    
    * move main.py to root
    
    * add packages
    
    * ruff fixes
    
    * remove unused gitignore and pre-commit-config
    
    * move ocpapi readme
    
    * remove extra github workflows from data/oc
    
    * remove extra pyproject.toml and setup.py
    
    * move enviornment deps to packages
    
    ---------
    
    Co-authored-by: apalizha <34039273+apalizha@users.noreply.github.com>
    Co-authored-by: Aini Palizhati <apalizha@andrew.cmu.edu>
    Co-authored-by: Brook Wander <brook.l.wander@gmail.com>
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: Janice Lan <janlan@fb.com>
    Co-authored-by: Janice Lan <janlan@meta.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Co-authored-by: Brook Wander <brookwander@devfair0771.h2.fair>
    Co-authored-by: Brook Wander <73855115+brookwander@users.noreply.github.com>
    Co-authored-by: Kyle Michel <kylemichel@meta.com>
    Co-authored-by: Kyle Michel <59006240+kjmichel@users.noreply.github.com>
    Co-authored-by: Brandon Wood <bmwood@fb.com>
    Co-authored-by: anuroopsriram <anuroop.sriram@gmail.com>
    Co-authored-by: anuroopsriram <anuroops@fb.com>
    Co-authored-by: Anuroop Sriram <anuroops@meta.com>
    Co-authored-by: Xiaohan Yu <78917162+tdytjd@users.noreply.github.com>
    Co-authored-by: Michael G. Taylor <119455260+mgt16-LANL@users.noreply.github.com>
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Daniel Levine <levineds@meta.com>
    Co-authored-by: Brandon Wood <b.wood@berkeley.edu>
    Co-authored-by: Daniel Levine <levineds@gmail.com>
    Co-authored-by: EC2 Default User <ec2-user@ip-172-31-1-84.us-west-2.compute.internal>
    Former-commit-id: 0ededf69a58ecea9f41a509a8f01f08784d591ef

[33mcommit 8f20f8ece4e4ae2f273e9fc6668ccfe9fe014884[m
Author: Nima Shoghi <nimashoghi@gmail.com>
Date:   Mon May 6 12:14:19 2024 -0400

    LBFGS batch size fix (#440)
    
    * General cleanup
    
    * Batchwise alpha/beta/rho computation
    
    * Refactors lbfgs step to not do an extra step after satisfying fmax
    
    * Gets back update_graph; commented from main loop
    
    * shape comment
    
    * bugfix for nondeterministic fmax
    
    * flatten energies per ocp2.0 changes
    
    * lt logic
    
    * update relaxation defaults
    
    * update relaxation hyperparm defaults
    
    ---------
    
    Co-authored-by: Nima Shoghi <nimashoghi@fb.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: 45779ead79cd383e9c405375c0278cd22afc6850

[33mcommit 8a003db9377432bc265c58cb464f5179db647c11[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri May 3 11:27:44 2024 -0600

    [BE] Upgrade all dependencies (#654)
    
    * enables building of basic package
    
    * enables building of basic package
    
    * use pyproject.toml only (no setup.py/cfg)
    
    * set package version dynamically from tags/commits
    
    * rm linting setup files and add ruff to .pre-commit
    
    * add GA workflow files
    
    * fix pytest adopts
    
    * fix pytest adopts
    
    * remove black keep ruff only
    
    * define pretrained models in yaml file
    
    * lint in GA instead of circleCI
    
    * rm python_lint job
    
    * add release.yml
    
    * install torch in test workflow (for now)
    
    * conditional requirements.txt
    
    * same issue with torch install
    
    * install torch-sparse yuck
    
    * install torch-sparse yuck
    
    * pyg annoyances
    
    * use requirements.txt
    
    * fix install requirements-optional.txt in ci
    
    * only what's used
    
    * set requirements versions
    
    * url in requirements-optional.txt
    
    * sneaky torch_cluster hidden in pyg functions
    
    * first stab at combined docs and tutorial
    
    * gh-pages push
    
    * reorg
    
    * first stab at combined docs and tutorial
    
    * gh-pages push
    
    * reorg
    
    * remove pandoc docs in favor of jupyter notebook autosphinx
    
    * move license
    
    * add pandoc
    
    * change install reqs
    
    * bump python in docs, add torch geometric/etc to install
    
    * test on push always
    
    * docs on release
    
    * on push (to test only now)
    
    * no os/python matrix
    
    * remove on push
    
    * add dependabot.yml
    
    * autoapi docs
    
    * small typos in links (many more to go)
    
    * rename README to index to make root url cleaner
    
    * add release to test pypi
    
    * remove old build workflow
    
    * test version of release.yml
    
    * update docs workflow
    
    * fix pypi name
    
    * use ocpmodels get_checkpoint, start fixing notebook running, execute notebooks by default
    
    * move tutorial utils to main repo, rename catalyst tutorial folders, add video links to site
    
    * add legacy tutorials
    
    * update kernels for legacy tutorials
    
    * more small fixes/edits
    
    * more small fixes to utils in tutorials
    
    * remove circleci
    
    * remove on push from release.yml
    
    * cleanup of models, fix for tutorial monkeypatch
    
    * docs workflow with call and dispatch
    
    * full release flow dry run
    
    * remove/add on push
    
    * no release on push
    
    * always lint
    
    * fix checkpoint links that are currently scaling factor files
    
    * small edits to model_registry
    
    * refactor model_registry.py to models
    
    * fix tutorial references to new ocpmodels.models.model_registry, and change old md files to links to documentation
    
    * more compact oc20 table for data download
    
    * all tests pass with torch 2.2 and pyg 2.5
    
    * [BE] Lint typing cleanup (#648)
    
    * ruff format everything
    
    * use new type hints
    
    * ruff fix with unsafe-fixes :-S
    
    * more ruff fixes
    
    * use find packages in pyproject.toml
    
    * code cleanup
    
    * more code cleanup
    
    * cleanup
    
    * more and more cleanup
    
    * cleanup top-level imports
    
    * always lint
    
    * last round
    
    * give it an abstractmethod
    
    * happy lints
    
    * update install docs
    
    * add demo link, quickstart jupyter blocks
    
    * more tutorial fixes to ocpmodels utils
    
    * broken link
    
    * no fastapi no black
    
    * only numpy pinned
    
    * make main.py run as cli `ocp`
    
    * clean up Runner and trainer context
    
    * keep main.py
    
    * lots of small fixes
    
    * no version pins :-S
    
    * gather objects helper function
    
    * many more small fixes
    
    * predict using gather and no more ragged arrays
    
    * more small fixes to install and legacy tutorial doc
    
    * use gather in run relaxations
    
    * all tests pass with torch 2.2 and pyg 2.5
    
    * only numpy pinned
    
    * no version pins :-S
    
    * gather objects helper function
    
    * predict using gather and no more ragged arrays
    
    * use gather in run relaxations
    
    * make main.py run as cli `ocp`
    
    * clean up Runner and trainer context
    
    * keep main.py
    
    * keep default logger as wandb
    
    * more small fixes
    
    * syrupy in dev only
    
    * simplify ApproxSnapshot to play friendly with new interface
    
    * more small fixes
    
    * update snapshots
    
    * more small fixes
    
    * add submitit as a dependency
    
    * enable notebook builds
    
    * typo in torchaudio
    
    * update trigger
    
    * extra-index
    
    * change order again for install
    
    * move to same install process as tests
    
    * gemnet for ocp-intro
    
    * fix doc trigger
    
    * reorg
    
    * add adsorbml and ocpapi examples
    
    * move license back to chapter
    
    * small fixes
    
    * more small fixes
    
    * more small fixes
    
    * remove gemnet nrr example
    
    * more small fixes to notebooks
    
    * more small fixes, remove monkeypatch as it's already in the code!
    
    * extend timeout, small fixes to embedding notebook
    
    * scikit-image instead of skimage, final calc.embed in embeddings tutorial
    
    * fix monkeypatch, optional visualization in adsorbml
    
    * move toc
    
    * add vdict to install, small fixes to monkeypatch
    
    * small fixes
    
    * NRR example change
    
    * small fix to inference, increase epochs for fine tuning
    
    * look for primary_metric in config, small fix to inference
    
    * include default primary_metric for s2ef
    
    * batch size 4 for smaller fine tuning
    
    * reduce epochs for fine-tuning
    
    * fewer epochs for finetuning experiments, fix data subset in inference
    
    * small fix to inference, tweak to timeout and batch size to make things faster
    
    * back to batch size 4
    
    * fix link to intro series
    
    * eval_every 10, fewer epochs to get fine-tuning under 30min
    
    * fix arg typo
    
    * allow distutils.py to run with torchrun
    
    * conditional check for config update, revert changes to update_config
    
    * small change for catalysis->chemistry
    
    * small typo in new_trainer_context
    
    * add GA tests status badge
    
    * docs on dispatch and call
    
    * adding CatTSunami tutorial
    
    * float32 in cosine_similarity
    
    * remove catsunami install for the moment
    
    * minor doc updates
    
    * torchrun and only ocp trainer
    
    * lmdbdataset only + typo
    
    * update readme with links
    
    * happy lints
    
    * more lints
    
    * remove more md files, exec time on cpu
    
    * add on push for now to see built docs
    
    * is this any different?
    
    * force lint config
    
    * set some minimum dependencies, update env.ymls
    
    * no more lint erros
    
    * update installation instructions
    
    * add pyg libraries to ymls
    
    * update doc readme
    
    * lint fixes
    
    * test _toc.yml
    
    * update example config
    
    * cleanup
    
    * revert toc test
    
    * revert example config
    
    * remove debug
    
    * remove isort
    
    * model_registry docstring
    
    * wandb default
    
    * explicit model names
    
    * update calculator tests to use model registry
    
    * update all header docstrings with Meta
    
    * fix typos in install docs
    
    * update snapshot
    
    * test only py 3.9 and 3.11
    
    * update snapshot!
    
    * darned snapshot
    
    ---------
    
    Co-authored-by: Misko <misko@meta.com>
    Co-authored-by: Zack Ulissi <zulissi@meta.com>
    Co-authored-by: Vahe Gharakhanyan <vaheg@meta.com>
    Co-authored-by: Brook Wander <brook.l.wander@gmail.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: 1e0afd2843d3e3e14362e6758d5069234a1db13f

[33mcommit 1e07c21350947e8bfa38420e4f1e7c291c00259b[m
Author: Misko <misko@meta.com>
Date:   Fri May 3 09:47:33 2024 -0700

    [BE] PyPi integration +  local model cache + Github actions (#623)
    
    * enables building of basic package
    
    * enables building of basic package
    
    * use pyproject.toml only (no setup.py/cfg)
    
    * set package version dynamically from tags/commits
    
    * rm linting setup files and add ruff to .pre-commit
    
    * add GA workflow files
    
    * fix pytest adopts
    
    * fix pytest adopts
    
    * remove black keep ruff only
    
    * define pretrained models in yaml file
    
    * lint in GA instead of circleCI
    
    * rm python_lint job
    
    * add release.yml
    
    * install torch in test workflow (for now)
    
    * conditional requirements.txt
    
    * same issue with torch install
    
    * install torch-sparse yuck
    
    * install torch-sparse yuck
    
    * pyg annoyances
    
    * use requirements.txt
    
    * fix install requirements-optional.txt in ci
    
    * only what's used
    
    * set requirements versions
    
    * url in requirements-optional.txt
    
    * sneaky torch_cluster hidden in pyg functions
    
    * first stab at combined docs and tutorial
    
    * gh-pages push
    
    * reorg
    
    * first stab at combined docs and tutorial
    
    * gh-pages push
    
    * reorg
    
    * remove pandoc docs in favor of jupyter notebook autosphinx
    
    * move license
    
    * add pandoc
    
    * change install reqs
    
    * bump python in docs, add torch geometric/etc to install
    
    * test on push always
    
    * docs on release
    
    * on push (to test only now)
    
    * no os/python matrix
    
    * remove on push
    
    * add dependabot.yml
    
    * autoapi docs
    
    * small typos in links (many more to go)
    
    * rename README to index to make root url cleaner
    
    * add release to test pypi
    
    * remove old build workflow
    
    * test version of release.yml
    
    * update docs workflow
    
    * fix pypi name
    
    * use ocpmodels get_checkpoint, start fixing notebook running, execute notebooks by default
    
    * move tutorial utils to main repo, rename catalyst tutorial folders, add video links to site
    
    * add legacy tutorials
    
    * update kernels for legacy tutorials
    
    * more small fixes/edits
    
    * more small fixes to utils in tutorials
    
    * remove circleci
    
    * remove on push from release.yml
    
    * cleanup of models, fix for tutorial monkeypatch
    
    * docs workflow with call and dispatch
    
    * full release flow dry run
    
    * remove/add on push
    
    * no release on push
    
    * always lint
    
    * fix checkpoint links that are currently scaling factor files
    
    * small edits to model_registry
    
    * refactor model_registry.py to models
    
    * fix tutorial references to new ocpmodels.models.model_registry, and change old md files to links to documentation
    
    * more compact oc20 table for data download
    
    * [BE] Lint typing cleanup (#648)
    
    * ruff format everything
    
    * use new type hints
    
    * ruff fix with unsafe-fixes :-S
    
    * more ruff fixes
    
    * use find packages in pyproject.toml
    
    * code cleanup
    
    * more code cleanup
    
    * cleanup
    
    * more and more cleanup
    
    * cleanup top-level imports
    
    * always lint
    
    * last round
    
    * give it an abstractmethod
    
    * happy lints
    
    * update install docs
    
    * add demo link, quickstart jupyter blocks
    
    * more tutorial fixes to ocpmodels utils
    
    * broken link
    
    * no fastapi no black
    
    * make main.py run as cli `ocp`
    
    * clean up Runner and trainer context
    
    * keep main.py
    
    * lots of small fixes
    
    * many more small fixes
    
    * more small fixes to install and legacy tutorial doc
    
    * make main.py run as cli `ocp`
    
    * clean up Runner and trainer context
    
    * keep main.py
    
    * keep default logger as wandb
    
    * more small fixes
    
    * more small fixes
    
    * more small fixes
    
    * add submitit as a dependency
    
    * enable notebook builds
    
    * typo in torchaudio
    
    * update trigger
    
    * extra-index
    
    * change order again for install
    
    * move to same install process as tests
    
    * gemnet for ocp-intro
    
    * fix doc trigger
    
    * reorg
    
    * add adsorbml and ocpapi examples
    
    * move license back to chapter
    
    * small fixes
    
    * more small fixes
    
    * more small fixes
    
    * remove gemnet nrr example
    
    * more small fixes to notebooks
    
    * more small fixes, remove monkeypatch as it's already in the code!
    
    * extend timeout, small fixes to embedding notebook
    
    * scikit-image instead of skimage, final calc.embed in embeddings tutorial
    
    * fix monkeypatch, optional visualization in adsorbml
    
    * move toc
    
    * add vdict to install, small fixes to monkeypatch
    
    * small fixes
    
    * NRR example change
    
    * small fix to inference, increase epochs for fine tuning
    
    * look for primary_metric in config, small fix to inference
    
    * include default primary_metric for s2ef
    
    * batch size 4 for smaller fine tuning
    
    * reduce epochs for fine-tuning
    
    * fewer epochs for finetuning experiments, fix data subset in inference
    
    * small fix to inference, tweak to timeout and batch size to make things faster
    
    * back to batch size 4
    
    * fix link to intro series
    
    * eval_every 10, fewer epochs to get fine-tuning under 30min
    
    * conditional check for config update, revert changes to update_config
    
    * small change for catalysis->chemistry
    
    * small typo in new_trainer_context
    
    * add GA tests status badge
    
    * docs on dispatch and call
    
    * adding CatTSunami tutorial
    
    * remove catsunami install for the moment
    
    * minor doc updates
    
    * torchrun and only ocp trainer
    
    * lmdbdataset only + typo
    
    * update readme with links
    
    * happy lints
    
    * more lints
    
    * remove more md files, exec time on cpu
    
    * add on push for now to see built docs
    
    * is this any different?
    
    * force lint config
    
    * no more lint erros
    
    * update installation instructions
    
    * update doc readme
    
    * lint fixes
    
    * test _toc.yml
    
    * update example config
    
    * cleanup
    
    * revert toc test
    
    * revert example config
    
    * remove isort
    
    * model_registry docstring
    
    * wandb default
    
    * explicit model names
    
    * update calculator tests to use model registry
    
    * update all header docstrings with Meta
    
    * fix typos in install docs
    
    ---------
    
    Co-authored-by: lbluque <lbluque@meta.com>
    Co-authored-by: Luis Barroso-Luque <lbluque@users.noreply.github.com>
    Co-authored-by: Zack Ulissi <zulissi@meta.com>
    Co-authored-by: Vahe Gharakhanyan <vaheg@meta.com>
    Co-authored-by: Brook Wander <brook.l.wander@gmail.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: 42c62a675834e2f0a199978f3b3da3a33081c4fc

[33mcommit 02d83f5fa70842575eb9759b046899ec6593ebbf[m
Author: Misko <misko@meta.com>
Date:   Thu Apr 25 08:48:57 2024 -0700

    add unique key loader check to YAML to avoid problematic configs (#658)
    
    
    
    Former-commit-id: 0abcc7e8c58b05a226663d004940f0726c172dd2

[33mcommit 20940f8d2805a31b6b4e2e3a6659e93fca489a91[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Fri Apr 19 11:55:27 2024 -0700

    do not add empty lines to include_dirs (#656)
    
    
    
    Former-commit-id: 4378ac4f73338af2767f7f51fc00ed98e27eeda4

[33mcommit 97ca9a7ff071aa8f8da6e35cdcfd4fabeb2baa7e[m
Author: Daniel Levine <levineds@gmail.com>
Date:   Thu Apr 11 17:10:16 2024 -0700

    Invert experimental import logic to opt-in (#651)
    
    * Invert experimental import logic to opt-in
    
    * remove default user import
    
    Former-commit-id: ee50cd140adbf09a41b329d31146d9c8cc037b6c

[33mcommit dbaefaed40eee2844d033c78ccd2fe68976ebcb6[m
Author: Janice Lan <janlan@fb.com>
Date:   Tue Apr 9 13:48:51 2024 -0700

    set wandb as default logger (#647)
    
    
    
    Former-commit-id: 2b9dad5dd79719ba1faf0de55b6640b746922442

[33mcommit 365b7758a414938cc242944ab46574eb38ee9259[m
Author: Misko <misko@meta.com>
Date:   Mon Apr 8 16:14:54 2024 -0700

    change _compute_metrics not to modify callers version (#645)
    
    
    
    Former-commit-id: 23b0e25c536c9e1b307aa552f7f8d0d8dfd444e3

[33mcommit e8a7abfd8ad66468328c6265b6f5673b06f6549a[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Apr 8 12:17:13 2024 -0700

    lin ref in ase datasets (#643)
    
    
    
    Former-commit-id: 5bb0a1df6c0df292af237ae347cc136bed149e35

[33mcommit c116f22ed5ad44cf5e833861bb3d411c240d1ed6[m
Author: Misko <misko@meta.com>
Date:   Mon Apr 8 11:44:28 2024 -0700

    add warning and optimize sampler per @lbluque suggestsion (#642)
    
    
    
    Former-commit-id: 01cce6c5a68321d76764e3cef38aec83971705fc

[33mcommit 5d0f59ebac5b2128142e75f1f1b72b276003b08c[m
Author: Misko <misko@meta.com>
Date:   Tue Apr 2 10:51:36 2024 -0700

    add step(batch) stateful sampler (#639)
    
    * add step(batch) stateful sampler
    
    * add option for no shuffling with test
    
    * lint
    
    * make sure that for the same seed we get the same orderings as the underlying pytorch smapler currently used
    
    * forgot to commit changes to data_parallel
    
    * lint
    
    * Simplify logic by calling super() as suggested by @mshuaibii
    
    Former-commit-id: 4eba45ca352ea89f6e3154629cbb90a76aa9c37f

[33mcommit cb00fb92419f79f3f1793cf518f0617666a3bba3[m
Author: Luis Barroso-Luque <lbluque@users.noreply.github.com>
Date:   Mon Apr 1 10:38:07 2024 -0700

    Ase dataset updates (#622)
    
    * minor cleanup of lmbddatabase
    
    * ase dataset compat for unified trainer and cleanup
    
    * typo in docstring
    
    * key_mapping docstring
    
    * add stress to atoms_to_graphs.py and test
    
    * allow adding target properties in atoms.info
    
    * test using generic tensor property in ase_datasets
    
    * minor docstring/comments
    
    * handle stress in voigt notation in metadata guesser
    
    * handle scalar generic values in a2g
    
    * clean up ase dataset unit tests
    
    * allow .aselmdb extensions
    
    * fix minor bugs in lmdb database and update tests
    
    * make connect_db staticmethod
    
    * remove redundant methods and make some private
    
    * allow a list of paths in AseDBdataset
    
    * remove sprinkled print statement
    
    * remove deprecated transform kwarg
    
    * fix doctring typo
    
    * rename keys function
    
    * fix missing comma in tests
    
    * set default r_edges in a2g in AseDatasets to false
    
    * simple unit-test for good measure
    
    * call _get_row directly
    
    * [wip] allow string sids
    
    * raise a helpful error if AseAtomsAdaptor not available
    
    * remove db extension in filepaths
    
    * set logger to info level when trying to read non db files, remove print
    
    * set logging.debug to avoid saturating logs
    
    * Update documentation for dataset config changes
    
    This PR is intended to address https://github.com/Open-Catalyst-Project/ocp/issues/629
    
    * Update atoms_to_graphs.py
    
    * Update test_ase_datasets.py
    
    * Update test_ase_datasets.py
    
    * Update test_atoms_to_graphs.py
    
    * Update test_atoms_to_graphs.py
    
    * case for explicit a2g_args None values
    
    * Update update_config()
    
    * Update utils.py
    
    * Update utils.py
    
    * Update ocp_trainer.py
    
    More helpful warning for debug mode
    
    * Update ocp_trainer.py
    
    * Update ocp_trainer.py
    
    * Update TRAIN.md
    
    * fix concatenating predictions
    
    * check if keys exist in atoms.info
    
    * Update test_ase_datasets.py
    
    * use list() to cast all batch.sid/fid
    
    * correctly stack predictions
    
    * raise error on empty datasets
    
    * raise ValueError instead of exception
    
    * code cleanup
    
    * rename get_atoms object -> get_atoms for brevity
    
    * revert to raise keyerror when data_keys are missing
    
    * cast tensors to list using tolist and vstack relaxation pos
    
    * remove r_energy, r_forces, r_stress and r_data_keys from test_dataset w use_train_settings
    
    * fix test_dataset key
    
    * fix test_dataset key!
    
    * revert to not setting a2g_args dataset keys
    
    * fix debug predict logic
    
    * support numpy 1.26
    
    * fix numpy version
    
    * revert write_pos
    
    * no list casting on batch lists
    
    * pretty logging
    
    ---------
    
    Co-authored-by: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: 24092fae39e1e45bec1795884b08218d47ccdb94

[33mcommit f91684527b59003da769c4269dbfc830678a2c2e[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Fri Mar 22 13:12:44 2024 -0400

    run_relaxation minor fixes (#636)
    
    
    
    Former-commit-id: 686f85039fca5ed74f45b664d1c1dde35b031726

[33mcommit e6ffc309469c05600f278d5e7ec195d8a497cfe2[m
Author: Misko <misko@meta.com>
Date:   Thu Mar 7 16:23:26 2024 -0800

    Add in tests for coefficient mapping, mprimary, lprimary (#626)
    
    
    
    Former-commit-id: 3e9234ab031954362e55be6db1814202550a20ff

[33mcommit 9bae91ff2b4676cd5d316763759af5fb2f36a30e[m
Author: Misko <misko@meta.com>
Date:   Mon Feb 26 09:31:16 2024 -0800

    add seed option for calculator (#624)
    
    * add seed option for calculator
    
    * add test setting seed in escn
    
    * break out tests to provide terminal output so circleCI does not time out
    
    Former-commit-id: 28f772a627450e88e6a08921fe91a71a2e37ffe0

[33mcommit ed63c8963427aeb2097906f920fa0185cf8dd13d[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Feb 14 08:27:39 2024 -0800

    Release ODAC23 data in extxyz format (#625)
    
    * Data release in extxyz format and config file fixes
    
    * Data release in extxyz format and config file fixes
    
    * Update DATASET.md
    
    ---------
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 447242f7a96898af60da878b022c9ed5913bd9c5

[33mcommit 6a481565d2bd8ba696361023d20f53c987361964[m
Author: Janice Lan <janlan@fb.com>
Date:   Tue Feb 13 16:32:44 2024 -0800

    Fix amp scale factor for loss (#617)
    
    * get amp scaler factor for loss before it gets updated in backwards
    
    * save metrics before applying scaler to loss
    
    ---------
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: cd549a9a8755b7ef3f219aa2ffb7c5b186a2e87d

[33mcommit 423c32120ef36c74aeb8b7d9abdb8ad73c164222[m
Author: Misko <misko@cs.toronto.edu>
Date:   Tue Feb 13 16:12:34 2024 -0800

    bump miniconda version (#627)
    
    
    
    Former-commit-id: 8b71ce245707aa953bf5451fb548912644f577de

[33mcommit ca363fe18d9ef0cc685c8b402c4c66dddbcab572[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jan 12 13:10:12 2024 -0800

    Set default val and test dataset class to be same as train (#618)
    
    
    
    Former-commit-id: e393068a8cf24a19a993afdf81018ffce1f99a05

[33mcommit 3ee9f30020cd68fae4abb37e7db9a32ee8df741e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jan 10 09:23:58 2024 -0800

    fix config spacing (#616)
    
    
    
    Former-commit-id: 10e353a2667f86d89429d108af83fa7810030f42

[33mcommit e4730af90d0f1cd0bc390c40ec137f543955901f[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jan 9 09:52:51 2024 -0800

    minor fixes (#615)
    
    
    
    Former-commit-id: c1b69f8c05e3a234fca1c8d9ecd2224a14337aff

[33mcommit b6c5590358b51c58edbb6ae5e3aca0ce6c9310fe[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jan 5 09:43:13 2024 -0800

    Fix edge_index check in make_lmdb_sizes.py (#611)
    
    
    
    Former-commit-id: c6c630cfea58b2c24abe84f9253690bb46863997

[33mcommit 27b715486328e045860e8ac6f451a19276114eb9[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jan 5 08:55:37 2024 -0800

    Unified OCP Trainer (#520)
    
    * initial single trainer commit
    
    * more general evaluator
    
    * backwards tasks
    
    * debug config
    
    * predict support, evaluator cleanup
    
    * cleanup, remove hpo
    
    * loss bugfix, cleanup hpo
    
    * backwards compatability for old configs
    
    * backwards breaking fix
    
    * eval fix
    
    * remove old imports
    
    * default for get task metrics
    
    * rebase cleanup
    
    * config refactor support
    
    * black
    
    * reorganize free_atoms
    
    * output config fix
    
    * config naming
    
    * support loss mean over all dimensions
    
    * config backwards support
    
    * equiformer can now run
    
    * add example equiformer config
    
    * handle arbitrary torch loss fns
    
    * correct primary metric def
    
    * update s2ef portion of OCP tutorial
    
    * add type annotations
    
    * cleanup
    
    * Type annotations
    
    * Abstract out _get_timestamp
    
    * don't double ids when saving prediction results
    
    * clip_grad_norm should be float
    
    * model compatibility
    
    * evaluator test fix
    
    * lint
    
    * remove old models
    
    * pass calculator test
    
    * remove DP, cleanup
    
    * remove comments
    
    * eqv2 support
    
    * odac energy trainer merge fix
    
    * is2re support
    
    * cleanup
    
    * config cleanup
    
    * oc22 support
    
    * introduce collater to handle otf_graph arg
    
    * organize methods
    
    * include parent in targets
    
    * shape flexibility
    
    * cleanup debug lines
    
    * cleanup
    
    * normalizer bugfix for new configs
    
    * calculator normalization fix, backwards support for ckpt loads
    
    * New weight_decay config -- defaults in BaseModel, extendable by others (e.g. EqV2)
    
    * Doc update
    
    * Throw a warning instead of a hard error for optim.weight_decay
    
    * EqV2 readme update
    
    * Config update
    
    * don't need transform on inference lmdbs with no ground truth
    
    * remove debug configs
    
    * ocp-2.0 example.yml
    
    * take out ocpdataparallel from fit.py
    
    * linter
    
    * update tutorials
    
    ---------
    
    Co-authored-by: Janice Lan <janlan@fb.com>
    Co-authored-by: Richard Barnes <rbarnes@umn.edu>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 1db56e0354a537f1b0f590a92a5b0e9b3952fa12

[33mcommit b154639d56bd93ec9774f5124bcaaadc6f3e4f3b[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Dec 5 23:04:41 2023 -0800

    DDEC release and IS2RS clarification for ODAC23 (#602)
    
    * Update MODELS.md & DATASET.md as per comments
    
    * Update DATASET.md
    
    Former-commit-id: 2a56dfc1e248b49364af587be3c5c72a7cca45e3

[33mcommit c26cf597148eb0c3f544e5221dfbea20e57a5f41[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Mon Nov 27 13:44:01 2023 -0500

    Add natoms to ASE data (#598)
    
    
    
    Former-commit-id: f49e5192edf93dae382df95448c72c0911387ae6

[33mcommit 55dd1e5aaa41bfc2be1c5080cde7ee283fa25276[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Nov 21 11:29:33 2023 -0800

    Update EqV2 31M ODAC checkpoint (#599)
    
    * Adds link to updated EqV2 31M checkpoint
    
    * Minor updates to bibtex entries
    
    Former-commit-id: 2cadc6342121a7e8fa1e395076980296a72bb039

[33mcommit f374610316605f73427c75b59d26e792a4f0bc9a[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Nov 3 17:14:49 2023 -0400

    Update odac data link (#595)
    
    
    
    Former-commit-id: dd8ae3e3704ce4aeccfc5710e7f5fb6a282c5813

[33mcommit d9ea77b63c95ed75975a34fcba9d6d71e30bf361[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Nov 1 23:29:07 2023 -0700

    Changes for ODAC Release (#588)
    
    * Changes for ODAC23 dataset release
    
    * Changes for ODAC23 dataset release
    
    * Changes for ODAC23 dataset release
    
    * Changes for ODAC23 dataset release
    
    * Changes for ODAC23 dataset release
    
    * Changes for ODAC23 dataset release
    
    * Changes for ODAC23 dataset release
    
    * Added Equiformer Energy Trainer and removed project in configs
    
    * black-ed the new files
    
    * Update DATASET.md
    
    Removed "total" from the ODAC section
    
    * Update DATASET.md
    
    * Update MODELS.md
    
    * Update base.yml
    
    * Update DATASET.md
    
    Added citation
    
    * Update MODELS.md
    
    Added citation
    
    * Update README.md
    
    ---------
    
    Co-authored-by: Anuroop Sriram <anuroops@meta.com>
    Former-commit-id: 0e4b0f4b9ccdbd6d58880d861ca7adebb5b84524

[33mcommit 22aeade0e9c0376e8c77ef8eeb92ba2b14e134ab[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Oct 6 11:00:52 2023 -0700

    Release EquiformerV2 checkpoint trained on OC22 (#586)
    
    * Release EquiformerV2 checkpoint trained on OC22
    
    * Adds note about OC22 energy prediction
    
    * Update
    
    * Minor
    
    Former-commit-id: 1554ab4cda9e7a20fb37e52ec64056dd08ccffed

[33mcommit 2d7239921b22f7ec42d9eae0517b12c652ce4af9[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Oct 4 09:56:00 2023 -0700

    Adds optional per-element linear reference coefficients to EquiformerV2 (#584)
    
    * OC22 dloader bugfix when using with linref: pyg objects have `x` and `y` but set to None
    
    * Updates EquiformerV2 model to optionally ship with linref coefficients
    
    * Some more details of how the linear ref energies are computed
    
    * Don't autocast to float16 for total energy predictions
    
    * debugging circleci
    
    Former-commit-id: d646682a50ca94fc7f0254bdba8988e51766efab

[33mcommit 42eb243008fe86daf0874fc22b17206d9d046c20[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Oct 4 08:53:31 2023 -0700

    Fixes nested YAML config loading from the ASE calculator (#585)
    
    * Fixes nested yaml config loading from the ase calculator
    
    * Install libarchive on mac builds
    
    * Revert "Install libarchive on mac builds"
    
    This reverts commit 528e3fee061315da20bd983754373a7908531d40 [formerly 301467fb1f1c27ca95665cff6e2480c64752234c].
    
    * debugging circleci
    
    * disregard coverage ci failures
    
    * update thresh
    
    * remove github checks
    
    * codecov debug
    
    * make codecov informational, not ci blocking
    
    * ci debug
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <mushuaibi@gmail.com>
    Former-commit-id: e963f2f3cdc15bddbd24d45159e90f9839224b48

[33mcommit de4a7a29eb2f99087a1f83c2d42370095f9161a5[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Fri Sep 29 20:10:34 2023 -0400

    Support IS2RE-Direct Training with ASE Read Datasets (#579)
    
    * Support IS2RE direct with ase read datasets
    
    * Make isort happy
    
    * Make isort happier
    
    * Add more ASE dataset tests
    
    * Adjust error message
    
    * Update TRAIN.md
    
    Former-commit-id: f69fcce7b8fceec1d1878ffbc1d3367835540695

[33mcommit c34ab67527e22944e679015e04fa6c490b6729ab[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Thu Sep 14 15:36:15 2023 -0700

    Support for passing in stats to Equiformer V2 model (#576)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 1cba827ed8a66c6eb02495ed7f28d1c2a7dcc62f

[33mcommit b56a1aa608cb91edd6bf9d4f95419a7ebb5ba5ac[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 14 14:53:44 2023 -0700

    Type fixes for using the MultistepLR scheduler (#578)
    
    
    
    Former-commit-id: f3ac5a0a78e3afdc2d2b742db9b1e9e4b16eb08e

[33mcommit e2e2a978549d4099e1c88b0ea060cd7185098a2f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 7 13:15:00 2023 -0700

    Adds an FAQ (#564)
    
    * Adds an FAQ
    
    * Update
    
    * FAQ update
    
    * Adds a couple of common GemNet issues
    
    * Adds note on overriding yaml params from command line (refs #533)
    
    * Link to FAQ from readme
    
    * Typos
    
    * More typos
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: aaa2de905b9b268e612a1c2fbcef6da79f7648d7

[33mcommit 84009dcf4e3ae53b5e442a0e910f26ab252e1dfe[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 7 12:26:21 2023 -0700

    GemNet-GP dense layer bugfix (fixes #573) (#574)
    
    
    
    Former-commit-id: 7ca42ec82d35e6c5f16d597deafd035b5e89a2a3

[33mcommit 9a1acb3e6ee4b1df16b1c42a96ffb418f056d76f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Sep 1 11:25:24 2023 -0700

    Adds flag to configure strict no. of nbrs or not in EquiformerV2 (#571)
    
    
    
    Former-commit-id: 737dc1fd812c13e6075f0ccff03aa458812fa6ad

[33mcommit ad507ef6000fbce5bfb634977373f25417942840[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 24 15:13:39 2023 -0700

    Fixes Codecov badge URL (#568)
    
    
    
    Former-commit-id: ee60c2a785dfaf05c802b208cb82cd68f17242a0

[33mcommit f655d2a800f123189dd2751b48844e0f746a0b17[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 24 12:31:56 2023 -0700

    Bugfix in loading checkpoints (#567)
    
    
    
    Former-commit-id: 4d220bc62fc2927c0b269de92f23aff147947415

[33mcommit 94eb14f7131b53b868c2ef00bcea7904d10bc381[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 24 11:33:08 2023 -0700

    Adds a unit test for using OCP models with the ASE calculator (#565)
    
    * Adds a unit test for using OCP models with the ASE calculator
    
    * Drop DimeNet unit tests
    
    * Adds a note on whether models are trained on adsorption or total energies
    
    * Bump up tolerance
    
    * Update snapshot
    
    * mac vs linux differences are annoying
    
    Former-commit-id: 69898a6c9f803aa9ea1a6459335d546492216150

[33mcommit 3c995ab22af62301abf43d02c903ea506f3b6953[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Aug 7 11:11:43 2023 -0700

    Typecast scheduler params to int + fix slurm experiment logging (#555)
    
    
    
    Former-commit-id: 4714d9e49e01357576f1f21e575a0c9815d40d2c

[33mcommit 5e4ea32b82763a38c2e8ee2e839199f7f210479b[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Wed Aug 2 18:49:43 2023 -0400

    Bug Fixes for Tutorial (#525)
    
    * Patch for ase datasets missing fid
    
    * Handle empty configurations of connect_args, select_args, and a2g_args
    
    * Use more meaningful sid and fid values
    
    * Fallback for non-numeric sid values
    
    * Update for new versions of numpy
    
    * More fixes for new numpy versions
    
    * Address review comments
    
    Former-commit-id: fb8332cd2a5ce0669be0e76eb13c9341f5751717

[33mcommit 4ab870446a421c3df2a70a9bf84148505fc236a6[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Wed Aug 2 09:58:13 2023 -0700

    More type annotations (#548)
    
    
    
    Former-commit-id: 3600b2f0670256e2c049f4c8fee2dd8b8bdc3c59

[33mcommit 720c35711b6735a024c470190651874ae6e87bd0[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Wed Aug 2 09:51:09 2023 -0700

    Use logging instead of print (#551)
    
    * More type annotations
    
    * Use logging instead of print
    
    Former-commit-id: 8e2b2e14cb343b8bd3d882ed2c299f83d0f27dee

[33mcommit f5dc0903eb74d2aefcfc7d68e90745f190fbed22[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Sat Jul 29 10:34:51 2023 -0700

    More type annotations (#546)
    
    
    
    Former-commit-id: 43e8974ab4f27b955e36624f2640a0c4dd8c74be

[33mcommit 2c4f4baf4d39628a93a5a5d39ad143e9074d66d3[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Fri Jul 28 19:02:10 2023 -0700

    Add more types, handle empty registries gracefully (#545)
    
    
    
    Former-commit-id: 8b2eed013b4dc259908c9b3b69a9047f6e45fe2e

[33mcommit 0910efb6229acb9928f64c47b15bd8cd2efcd9e7[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Thu Jul 27 17:45:47 2023 -0700

    More type annotations (#544)
    
    
    
    Former-commit-id: 2617df947ae47450bd78c44ee5fa95b4c24596e1

[33mcommit ac93390c8dd9a6be8d54e84d73b0a39dc4ebf02a[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Mon Jul 24 17:41:21 2023 -0700

    Add autoflake to pre-commit to eliminate unused imports automatically (#534)
    
    
    
    Former-commit-id: 54e93922dfee04aa5785194d9eeb5aec2cf2f4fd

[33mcommit dcf1645cc5dee5d37cec2593965150a1a44b76da[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jul 24 09:44:50 2023 -0700

    Skipping scheduler setup in EquiformerV2 trainer if train_loader isn't defined (#541)
    
    
    
    Former-commit-id: 39e8bdd7277b4855eecd0a484e23d48e14a75009

[33mcommit f4e19b0c1a8e84ad1c6a34a8bd8d9da200705da1[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Thu Jul 20 12:20:56 2023 -0400

    Remove unused imports and sort imports (#528)
    
    
    
    Former-commit-id: 4e69123d2e0167ab1b25aabc5c2d38e237fcfed1

[33mcommit 8d9274b263b88ed8afcbdd7a64ab01cfea3bb63f[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jul 19 17:17:34 2023 -0700

    ocp calc amp fix (#529)
    
    * ocp calc amp fix
    
    * support non-amp ckpts
    
    * cleanup arg
    
    Former-commit-id: ec49b4c32d51dbae019cdaadb25656ac71e69d7a

[33mcommit 4daa26edc77001d942556772b3ebba7f36d556b1[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Wed Jul 19 11:33:30 2023 -0400

    Add isort to CI (#527)
    
    * Sort imports
    
    * Add isort to CI
    
    Former-commit-id: a97cf962817c6c257932daec5832b83dcf0d769e

[33mcommit 4617e78889e07684c0b80e23bc6099c4fa167eac[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jul 18 15:11:13 2023 -0700

    EquiformerV2 support (#518)
    
    * Support for running EquiformerV2 from within ocpmodels
    
    * Correct config urls
    
    * Pointers to the model weights
    
    * Adds a copy of the EquiformerV2 code to OCP
    
    * Adds e3nn dependency
    
    * Adds EquiformerV2 test
    
    * Snapshot update + minor config update
    
    * Update config urls
    
    * Does updating mac resources fix it?
    
    * Truncate to 2 layers
    
    * Cmon tests
    
    * Remove commented import line
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: ae6d53b4386b84efea752084c215e06868bab05f

[33mcommit 2ea245ef88e92af7002066af1805120baebf3f5d[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jul 11 16:55:59 2023 -0700

    escn cpu fix (#523)
    
    
    
    Former-commit-id: b9b67519d99242878a33a5d9f84385ec8e89e82c

[33mcommit b3495c626fa538a9fee5e7c82c0ec9404745f78e[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Mon Jul 10 17:51:06 2023 -0400

    Add type annotations (#522)
    
    
    
    Former-commit-id: 1a0897d746b26cfadfb428a5712c7e10d4d64a5f

[33mcommit a1edf0f0ad6b1d571695ecdf390406e1e24a1e91[m
Author: Richard Barnes <rbarnes@umn.edu>
Date:   Mon Jul 10 14:18:03 2023 -0400

    Add type annotations (#519)
    
    
    
    Former-commit-id: 8d7d3ad7e9b158aa2abe8675ddc6d5e7fc44a361

[33mcommit 454f0cfc9801228c6b1a1912b1f809d83b48e41e[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Thu Jun 15 13:00:55 2023 -0700

    LMDB-based ASE dataset, generic targets, target metadata guessing (#513)
    
    * commit for ase dataset
    
    * typo in sid
    
    * address reviews
    
    * address reviews
    
    * remove lock file for readonly lmdb database
    
    * address review
    
    * More comments
    
    * lint
    
    * address reviews
    
    * Apply suggestions from code review
    
    Co-authored-by: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
    
    * lint
    
    ---------
    
    Co-authored-by: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
    Former-commit-id: b492a658dbd15d1bd4611935037f033491bb34fd

[33mcommit cef85ad85be6cf869ddb99e5c62075f396e9997a[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Tue Jun 6 13:16:41 2023 -0400

    Add ASE Datasets (#492)
    
    * Add AseReadDataset
    
    * Add ASE Reader dataset to __init__
    
    * Fix typo
    
    * Adjust docstring
    
    * Fix typo
    
    * Add ASE DB Dataset
    
    * Lint
    
    * Adjust A2G settings
    
    * Fix typo
    
    * Bug fixes and consistent naming
    
    * Bug fixes and adjustments
    
    * Add handling for structures with no tags
    
    * Add tests of ASE datasets
    
    * Lint
    
    * Test __getitem__ methods
    
    * Default to not pre-compute graph on cpu
    
    * Adjust pattern-matching syntax
    
    * Update test
    
    * Allow user to specify arguments for ase.io.read
    
    * Bug fix
    
    * first commit
    
    * lint tests
    
    * hook lmdb into ase dataset
    
    * fix lmdb ase hook into dataset
    
    * Bug fix
    
    * Update env.common.yml
    
    Add orjson to conda dependencies
    
    * lint
    
    * Improve speed of ASE DB Datasets that use LMDB backend (#499)
    
    * Include structure with fixed atoms in test
    
    * fix constraints
    
    * Test if conda installs break with new cache
    
    * Update config.yml
    
    * Update config.yml
    
    * add metadata and tests
    
    * add metadata for lmdb datasets
    
    * add metadata for lmdb datasets
    
    * change metadata getter
    
    * Don't cast jagged list to np.array
    
    * Force rebuild circleci conda env
    
    * Remove comment
    
    * Suppress another warning on torch.tensorizing a list of np.arrays
    
    * Make untagged atoms check more explicit
    
    * Refactor and add AseMultiReadDataset
    
    * Lint
    
    * Refactor apply_tags to atoms_transform, add test of AseReadMultiStructureDataset
    
    * Remove lmdb lock file
    
    * Add test for ASE DB with deleted row
    
    * Lint
    
    * Fix broken filepath in test
    
    * Include sid values
    
    * Revert "Merge branch 'ase_lmdb' into ase_read_dataset"
    
    This reverts commit d98ecb7cd18d1ed9c5475d652fcfd14dde1c712b [formerly e7e791287c91cd3897f157087e412dd978039f42], reversing
    changes made to 79058339c399de85f98974f78450642182876a01 [formerly a87040de2af647d9ee0ed66a50eb321e30d1b9d4].
    
    * Address review comments
    
    * Document ASE datasets
    
    ---------
    
    Co-authored-by: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
    Co-authored-by: Zack Ulissi <zulissi@meta.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 8847cc227eb5a8570471f8968ac64d9e451928f9

[33mcommit 3a498b06ab72913ce1560031ba29934261d56aea[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Thu Jun 1 14:39:16 2023 -0700

    minor change (#509)
    
    Co-authored-by: Brook Wander <brookwander@devfair0771.h2.fair>
    Former-commit-id: b3df31a2a852e81df3e12c56274dd4b763b05d2e

[33mcommit 084ca614a20286077841882d6b4c9f1fcd3b1642[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed May 31 15:32:18 2023 -0700

    Maintenance updates (#506)
    
    * Version bump for ase, pymatgen
    
    * Gh action for closing inactive issues and PRs
    
    Former-commit-id: 37ecf41ee950ee76da316efbae96b2b4d45f4e32

[33mcommit 7bba8c4237725552f253011c7be1f6cef1760646[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Tue May 23 10:36:58 2023 -0700

    Fixes #501: don't cast a jagged list as np.array (#500)
    
    * Test if conda installs break with new cache
    
    * Update config.yml
    
    * Update config.yml
    
    * Don't cast jagged list to np.array
    
    * Force rebuild circleci conda env
    
    * Remove comment
    
    * Suppress another warning on torch.tensorizing a list of np.arrays
    
    ---------
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: cfc5f1759d8ebb458575292e7219d09204d60b9b

[33mcommit 76a707fbd05bc70b736a45ad15cfe82257051def[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu May 18 09:14:49 2023 -0700

    Adds GemNet-OC checkpoint trained on OC20+OC22 total energies with `enforce_max_neighbors_strictly=False` (#495)
    
    * Adds GemNet-OC ckpt trained on OC20+OC22 total energies with degenerate edges
    
    * Typo
    
    * Update MODELS.md
    
    Former-commit-id: 7c5dc3fa38e736d8bf822728d5734c03f5d54afb

[33mcommit ff0df7e2d8b7be2ebb46f7eb5ad1be8bb44e4002[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri May 5 19:02:10 2023 -0700

    Adds __init__ files to enable portability with `pip install` without the `-e` flag (#490)
    
    * Add a bunch of missing init files
    
    * Skip importing Jd till SCN needs to be run
    
    Former-commit-id: 7edaf26315fd987b12736847f3243abceb3d7257

[33mcommit a9fd45d543989ccc4596b0e2282b0095a1e1eaa7[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Tue May 2 18:18:47 2023 -0400

    Allow models to include degenerate edges (#467)
    
    * Allow for inclusion of all equidistant edges instead of breaking ties heuristically
    
    * Add tolerance to effective threshold
    
    * Compute max masking on per-atom basis
    
    * Allow symmetry enforcement in get_max_neighbors_mask via global cutoff
    
    * Allow different pooling functions for effective radii
    
    * Change terminology to clarify global pooling
    
    * Fix variable name
    
    * Remove global cutoff option
    
    * Allow edge degeneracy inclusion in GemNet-OC
    
    * Remove unneeded import
    
    * Fix typo
    
    * Lint
    
    Former-commit-id: 3a8ac143a53cce22422fa601e433ec5d31358f3c

[33mcommit 445153e53c08924f802d3fe871bdf343ee93e1d6[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Apr 25 19:21:24 2023 -0400

    Revert "LMDB folder fix (#476)" (#480)
    
    This reverts commit a4b5f86e43a8fe9ce2f272f4a8c90b4731d4013c [formerly 4c20e872734e72459edf3c3ac05e44f83c2e935b].
    
    Former-commit-id: f6c036b146d7ca55cd2d9e6cd13b57f2d04294ff

[33mcommit a4b5f86e43a8fe9ce2f272f4a8c90b4731d4013c[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Tue Apr 25 12:53:56 2023 -0700

    LMDB folder fix (#476)
    
    * first stab at lmdb len fix
    
    * slight refactor so that the encode works correctly
    
    * add readahead as default for lmdb, add some more docs to the lmdb dataset class
    
    * same changes for oc22 dataset objects as for the default LmdbDataset
    
    Former-commit-id: 4c20e872734e72459edf3c3ac05e44f83c2e935b

[33mcommit 5505aaf53adc9d2669ff26115bf423867e365996[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Mar 30 04:41:56 2023 -0700

    escn cpu support (#465)
    
    
    
    Former-commit-id: d09cd832defe4a204ac42ab35f2438f00e1be4c6

[33mcommit a2f7d85520e330b8707e8b7cb39528750d2bbca4[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Mar 16 09:05:09 2023 -0700

    eSCN checkpoints+configs  (#457)
    
    * upload escn checkpoints + configs
    
    * spacing
    
    Former-commit-id: 11c76863002f4bc088aeb4d042ce284ab8e6d78a

[33mcommit 3ceb6c3c5491c28e7ea89d7ee3554539440249e4[m
Author: Brandon Wood <bmwood@fb.com>
Date:   Mon Mar 13 17:23:05 2023 -0700

    Updating flags and docs for training on total energy targets (#455)
    
    * removed total_energy flag using oc20_ref instead
    
    * add train_on_oc20_total_energies flag and update docs
    
    * updated docs and added some asserts for more descriptive errors
    
    ---------
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: dc762df1d04f07a35e3e4f64a4ca425bfb3cf701

[33mcommit 2fd4442317b640ebdd797310d677e8ba7dd2680b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Mar 7 09:30:35 2023 -0800

    Updates GNOC-All-MD checkpoint and moves this and SCN checkpoints to /models from /data on S3 (#456)
    
    
    
    Former-commit-id: a378c350f83b7fcf5d64bdc86e6cc67bc520c846

[33mcommit f83d150b5fcb940b814f25043fb00914945b8708[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Feb 27 13:17:30 2023 -0800

    SCN checkpoint release + missing AdsorbML checkpoints (#453)
    
    * scn checkpoint/config release
    
    * organize scn configs
    
    * all->2M config
    
    * epochs->steps
    
    * upload checkpoints
    
    * Update MODELS.md
    
    * add GemNet-OC MD checkpoint
    
    * add # gpus used for training
    
    Former-commit-id: 7200399a57e410d0d8e84d30c7c0335c822e0179

[33mcommit 78b2a267185801fff0284d54696c34008982b95a[m
Author: zulissimeta <122578103+zulissimeta@users.noreply.github.com>
Date:   Tue Feb 21 22:02:14 2023 -0800

    Warnings for missing scale files during validation (#450)
    
    * Warnings for missing scale files during validation
    
    If running training without explicitly setting scale files, training will start and ensure_fitted will send warnings about unfitted scale parameters. However, during the validation step ensure_fitted throws hard errors about scale files being unfitted.
    
    This tiny PR changes the validate step to only throw warnings in this scenario, similar to the training loop.
    
    * Suppress error from `predict`
    
    * pin syrupy
    
    ---------
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: d51a9be745e0e0315880699b846acea48dfd535a

[33mcommit 1cee5b784505f21240ec46b5aa344b2b17a94e75[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Feb 14 17:30:36 2023 -0800

    fixes save-full-traj call (#451)
    
    
    
    Former-commit-id: 1f935a2d28935c4d689e9ba944c5a23e8bc26de6

[33mcommit 851a14a58a667621d2c5608776bef054768ce65c[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Feb 13 10:59:56 2023 -0800

    escn lint
    
    
    Former-commit-id: b8a203207533f8723b3a3b1b8560fea94bc2dc3a

[33mcommit 51479b739a8172557ad0afca6e8fe9bfe39f222a[m
Author: Larry Zitnick <zitnick@devfair0331.h2.fair>
Date:   Mon Feb 13 10:13:39 2023 -0800

    Adding eSCN model https://arxiv.org/abs/2302.03655
    
    
    Former-commit-id: c3a38f1f08e0817288fa915de651d78090d8681c

[33mcommit e1f97bbcc6e132276454a76b28d1b1735df186ab[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Feb 9 09:53:47 2023 -0800

    Catch OOM during relaxations (#447)
    
    * Dynamic batching
    
    * remove debug calls
    
    * move recovery outside except
    
    * revert local changes
    
    * cleanup
    
    * wrap only run in try/except block
    
    * extract batch size from sids
    
    * sid->natoms
    
    ---------
    
    Co-authored-by: anuroopsriram <anuroops@fb.com>
    Former-commit-id: 0b44ba89f40ef3af33b56b978ecd22544204786c

[33mcommit 332e661694cee45842793677fcc66d3146f039ae[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Feb 2 11:06:58 2023 -0800

    Support dataset sharding (#446)
    
    * add sharded support
    
    * update example config
    
    ---------
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 9f380ffc4b31282d05d92f712a6dbc35ec2984b7

[33mcommit 6116511973eef437dfbba409e81edab08e230997[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Feb 2 09:38:57 2023 -0800

    Relaxation storage optimizations (#445)
    
    * relax storage optimizations
    
    * remove writing mask
    
    * add mask
    
    * update documentation
    
    * cleanup, fix 1-idx bug
    
    * rm comment
    
    * add flag to example config
    
    * unwrap model call
    
    ---------
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 4b768d67763d8bfb34523546e16e0e19c3b786a5

[33mcommit 9a6fe189867a443802e120faa0160b54139f4784[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Feb 2 12:40:38 2023 +0530

    Upgrades to pytorch 1.13, cuda 11.6, pyg 2.2.0 and conda-only install (#442)
    
    * Updated deps
    
    * Typo
    
    * Drop windows
    
    * Get back codecov orb
    
    * Missed activating conda env
    
    * mamba install following Zack's suggestion
    
    * Getting back mamba in circle config
    
    * Updated instructions
    
    Former-commit-id: 7a6b5d767c416d5302e5ff7cf6d5fb2e7718afb5

[33mcommit 1de787ed54b9e7f8fb28ffea8724d65dfd2083f7[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Jan 30 21:16:31 2023 -0800

    Fixes #436: tag description in tutorial notebook (#444)
    
    
    
    Former-commit-id: 88d6fb043c6fb76e749c7b8b83e899476d43857f

[33mcommit f8b15c50b09d57b4fb5e9c7edcde6032d50972c3[m
Author: Brandon Wood <bmwood@fb.com>
Date:   Tue Jan 24 14:38:56 2023 -0800

    Float32 precision and EvalAI submission files for OC22 total energy predictions (#421)
    
    * write tot_e predicts in float32
    
    * adding total_energy=True to base OC22 configs
    
    * assert oc22 predictions are fp32
    
    * update to method for writing predictions to keep track of precision
    
    * submission file to support oc22
    
    * move energy values to cpu before writing predicts and updated make_submission script
    
    * minor fix
    
    * minor fix
    
    * update to include prediction_dtype flag and remove check in make_submission_file.py
    
    * added documentation for the prediction_type flag and oc22 evalai
    
    * updated oc22 docs in TRAIN.md and minor changes to make_submission_file.py
    
    * add joint training documentation
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: dd3d121e2cc383e28cc0d11cb84ae45a3154a397

[33mcommit a13cc8a713a93aee38c6ccf2974731e6ea522906[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jan 24 10:46:57 2023 -0800

    Zero out NaNs, if any, in the loss (#441)
    
    
    
    Former-commit-id: 0d81ae7e396361f9ef73f15eae1ac3d89fb23e59

[33mcommit 2e499714ed5103cf3f1bed90a8e4513247039310[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Jan 19 13:58:27 2023 -0800

    Support for using tag weights with L2MAE force loss (#439)
    
    
    
    Former-commit-id: 55308490eeb78fad72ac6fbbf189da79d6492065

[33mcommit dc1e8a31228d55360e8532cc7a33909c21c49bde[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jan 11 11:49:42 2023 -0800

    Use "deterministic" scatter for relaxations on CUDA tensors if available (#438)
    
    * Force scatter operations on GPU to be deterministic
    
    * Fix device bug
    
    * Default to false and adds note
    
    * Add config flag for deterministic scatter
    
    Former-commit-id: 1ad2d010f2214f29b1f259643e1468493eac4e79

[33mcommit 604a849637e7fa73075ba4f9189b51f0827aceaa[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Nov 9 14:15:50 2022 -0800

    Fixes #424: CPU/GPU device support with `OCPCalculator` (#430)
    
    
    
    Former-commit-id: 81ee786b7f2ff89076912822c719cd7e4fdfcff2

[33mcommit 0feb371e29eb353d90e5f6e0c6edfa47eb38e0cb[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Mon Nov 7 20:57:05 2022 -0500

    Make OCPCalculator use PBC configuration from ASE atoms objects (#429)
    
    * Make OCPCalculator use PBC configuration from ASE atoms objects
    
    * Update documentation in AtomsToGraphs
    
    * Fix typo
    
    * Move operations to torch
    
    * Change property to torch tensor
    
    * Update utils.py
    
    * Fixes for PBC handling
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: c31f197b3d7094a84cedea07923d4c5ace9cb976

[33mcommit b7750d6e3e19b9283cc6cd9fed51a4b1f7ddaa65[m
Author: Zachary Ulissi <zulissi@gmail.com>
Date:   Mon Nov 7 16:53:55 2022 -0500

    Update `dimenet_plus_plus` to use PyG activation function resolver (#427)
    
    * Update dimenet_plus_plus.py
    
    * update swish to silu
    
    Former-commit-id: 3e6128d5e94cf90f182767c66f993961539c7d8a

[33mcommit db1590c7c0f52f37a29c5324bcea0d841bee2ff7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 6 15:07:09 2022 -0700

    Updates bibtex entries for GemNet-OC, SCN, PaiNN (#420)
    
    
    
    Former-commit-id: e83253de5d6eb0bfcde5c96b2b5eadba5d6651ae

[33mcommit c4d832a82ad63f06c489eaacff080bdd85a8c500[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 4 08:35:04 2022 -0700

    Package scaling dictionary within GemNet-dT pretrained checkpoints (#419)
    
    
    
    Former-commit-id: 1299ff265933f1226bac2814394395b9bc442c87

[33mcommit ff25d66a3d24dbf8159611c70188bdbcf12a76e4[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Sep 30 14:11:21 2022 -0700

    OCPCalc edge cases (#417)
    
    * modify attr if avail
    
    * fix for cps with relax datasets
    
    * abs scale file path
    
    * nope
    
    Former-commit-id: 17dcc4ae7aecaaa65aa0a2c82771b86b72c49491

[33mcommit 5978875ea39136d7f138c2e583fc39960e1e0f6b[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Sep 28 12:44:46 2022 -0700

    alt trainer identifier (#416)
    
    * alt trainer identifier
    
    * clarify comments
    
    * remove reliance on description
    
    * reorder
    
    * indent fix
    
    * rm warn
    
    Former-commit-id: ba2605011d9966942ec15c9ead7225835c65681b

[33mcommit 826d7174c9cfeb17bbec0d74fa2638c07fb66a21[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Wed Sep 28 10:54:13 2022 -0400

    Include more thorough tests in test_radius_graph_pbc (#410)
    
    * Support PBC in all three dimensions in radius_graph_pbc
    
    * Support PBC flag for each direction
    
    * Add test of radius_graph_pbc
    
    * Change default PBC handling in radius_graph_pbc to [True, True, False]
    
    * Update radius_graph_pbc test
    
    * Update test_radius_graph_pbc.py
    
    * Add bulk test for radius_graph_pbc
    
    * Update test_radius_graph_pbc.py
    
    * Ensure parity between radius_graph_pbc and radius_graph
    
    Expands the unit test for `radius_graph_pbc` to ensure that `ocpmodels.common.utils.radius_graph_pbc` gives the same results as `torch_geometric.transforms.radius_graph` when called with PBC disabled.
    
    * Ensure offsets are zero if PBC disabled in radius_graph_pbc
    
    * Add small molecule test for radius_graph_pbc
    
    * More thorough tests in test_radius_graph_pbc
    
    Co-authored-by: Ethan Sunshine <esunshin@andrew.cmu>
    Former-commit-id: e90dc2745291b6f91c02364a14947e6187dbab6f

[33mcommit caee912ef79eebdea0d418f9afa01a561fa8a72f[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Wed Sep 28 10:33:18 2022 -0400

    Balanced batch sampler fix (#406)
    
    * Updated `BalancedBatchSampler` to ignore incompatible datasets
    
    * Add warning message when loading incompatible dataset
    
    * Better warning message
    
    * If balancing is disabled, we don't have to check `self.balance_batches` every iteration.
    
    * Allow boolean values for optim.load_balancing
    Code cleanup
    
    * Removed code duplication for balanced batch sampler
    Added tests for balanced batch sampler
    
    * Updated tests to pass on Windows
    
    * Fix failing windows tests
    
    
    Former-commit-id: cfe706b0c30d4735cc431aa3d310dc06766f124d

[33mcommit a2853140d2943ee79f5480ce44a111da858ff427[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Sep 28 06:44:51 2022 -0700

    OC22 pretrained models (#414)
    
    * gemnet-oc oc22 configs
    
    * add oc22 s2ef model checkpoints
    
    * prefix https
    
    * minor fixes
    
    Former-commit-id: 1780222e056f6b27d9d20840d3476ff18b666023

[33mcommit 3088e91fe4856e474128e7f262f9c570bc13e644[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Sep 27 12:17:43 2022 -0700

    Reduce CircleCI MacOS resource class (#415)
    
    
    
    Former-commit-id: 2b83096a975dd3c32f92989c0ab791428cf19caf

[33mcommit 61d0fd01b10212e08bdb36c9dc7844e69c2a6797[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Thu Sep 22 13:52:11 2022 -0400

    Remove Python 3.8 only fstring interpolations (#411)
    
    
    
    Former-commit-id: 983c39ef5887d583407dd33b9232b7aad0427162

[33mcommit a2e3b9b985663f63a767407612716d753e113e42[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Mon Sep 19 16:12:56 2022 -0400

    Fix pre-empt recovery crash (#407)
    
    * Fix issue with config not containing model name when recovering from a pre-empt
    
    * Fix lint issue
    
    Former-commit-id: d3c8112480eb868f30ddfc759526c0bce43f3311

[33mcommit 3d6ab604edb10257ac725a9fce869f2771c0e801[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Tue Sep 13 18:21:41 2022 -0400

    Consistent scaling module across models, stricter consistency checks (#390)
    
    * Moved all trainer context setup code to `new_trainer_context` function
    
    * Updated `BaseTrainer.save` to return the saved checkpoint path
    
    * Added scaling module for creating scale factors that are stored in the model checkpoint as buffers
    
    * Updated GemNet to use new scaling module
    
    * Updated GemNet-GP to use new scaling module
    
    * Updated GemNet-OC to use new scaling module
    
    * Updated PaiNN to use new scaling module
    
    * Added backwards compatibility with old `.json` and `.pt` scaling modules
    Fixed bug with GemNet and GemNet GP
    
    * Updated backwards compatibility to work with checkpoints with training state
    Updated `fit.py` script to allow for updating old checkpoints to new format
    
    * Support explicit scale factors dicts
    
    * Updated `load_scales_compat` to patch the model's `load_state_dict` to ignore missing scale factors
    
    * Format base_trainer
    
    * Reverted `load_state_dict` patching: This didn't work because `load_state_dict` isn't called on child modules
    Created our own `load_state_dict` wrapper util that ignores scale factor keys
    
    * Added consistency check when scale factors are loaded multiple times (e.g., when scale factors are loaded using both traditional and new methods)
    
    * Added error when `scale_file` doesn't exist
    Added warning for empty scale dict
    
    * Added support for checkpoints with the "scale_dict" key
    
    * Added check for unfitted scale factors before train/val/predict/run-relaxations
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 366803fb582f8fbc381c996abab3809904bb7d48

[33mcommit 7cb574370129586e40cf1a83e1ce75d0f9cb7cde[m
Author: Ethan Sunshine <93541000+emsunshine@users.noreply.github.com>
Date:   Thu Sep 8 15:17:25 2022 -0400

    Support PBC in all three dimensions in radius_graph_pbc (#394)
    
    * Support PBC in all three dimensions in radius_graph_pbc
    
    * Support PBC flag for each direction
    
    * Add test of radius_graph_pbc
    
    * Change default PBC handling in radius_graph_pbc to [True, True, False]
    
    * Update radius_graph_pbc test
    
    * Update test_radius_graph_pbc.py
    
    * Add bulk test for radius_graph_pbc
    
    * Update test_radius_graph_pbc.py
    
    * Ensure parity between radius_graph_pbc and radius_graph
    
    Expands the unit test for `radius_graph_pbc` to ensure that `ocpmodels.common.utils.radius_graph_pbc` gives the same results as `torch_geometric.transforms.radius_graph` when called with PBC disabled.
    
    * Ensure offsets are zero if PBC disabled in radius_graph_pbc
    
    * Add small molecule test for radius_graph_pbc
    
    Co-authored-by: Ethan Sunshine <esunshin@andrew.cmu>
    Former-commit-id: 0d9635aa14e342bf482f7890b1899f73d871cfa3

[33mcommit afba608302236c533c2f865e589bea155323d3b7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Sep 7 16:44:06 2022 -0700

    Integrates codecov (#403)
    
    * Trying out codecov
    
    * pip install pytest-cov
    
    * Codecov badge
    
    Former-commit-id: a4885128219f61a764b2cc4041e5e9daa11151fb

[33mcommit 3e617095b7a1d81d7b593a0c2f3765c4f3f2ce5c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Sep 7 16:33:25 2022 -0700

    Explicitly check e3nn == 0.2.6 when running SCN (#404)
    
    
    
    Former-commit-id: 7f553b1bf5c97d7024398cc31b5443bdce31462f

[33mcommit cc0971d563a45102f8b2ac6202262fffca5aec84[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Sep 7 14:12:22 2022 -0700

    Revert to older / faster implementation of `wigner_D` in e3nn.o3 (#402)
    
    * Revert to older implementation of `wigner_D`
    
    * Pointer to 0.5.0 change
    
    Former-commit-id: 69bd94152033c4acc7f4b7666aa8a50de7ea1afb

[33mcommit e7a3705b628f66ae573abf41718e4857d20158a5[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Wed Sep 7 15:00:26 2022 -0400

    Build and tests on macos and windows (#401)
    
    * Build and test on macos and windows as well
    
    * Make all model tests use the registry
    
    * Update
    
    * Update
    
    * Update
    
    * Update
    
    * Update
    
    * Updated `setup_imports` to work with Windows
    
    * Decrease default tolerances so tests pass on Windows
    Remove atol/rtol from serialized approx representation
    
    * Fix excpetion during import error construction when registry is empty
    
    * Fix snapshot parse error
    
    * Upgrade `certifi` on Windows to fix SSL issues
    
    * Install/update `python-certifi-win32` on Windows as well
    
    * Try using `pip-system-certs` package to fix SSL issues
    
    * Trigger CI build without invalidating cache
    
    * Change checkpoint loading to use requests module
    
    * Read checkpoint bytes into io.BytesIO before loading
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 5dfabf4dd0ea1952f188949769bc733b753a1523

[33mcommit 6ffb63df0bd8f3907a83e5c45c5bf737d7571fb6[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Tue Sep 6 20:43:41 2022 -0400

    Snapshot testing using syrupy (#399)
    
    * Added syrupy as a dependency
    
    * Updated model tests to use snapshot testing (using syrupy)
    
    * Cleaned up unused imports in tests
    
    * Added a syrupy extension for properly handling approx comparisons of numpy arrays
    Updated model tests and snapshots to match this new format
    
    * Decrease default atol to 1e-6
    
    * Decrease rtol/atol for ForceNet
    
    * Added more comments and license
    
    Former-commit-id: 8d1016b37e3d3e2344ab1221684da20bef8fb483

[33mcommit 5da993c1501d4d5e88d35799284043d7c01a5dcf[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Sep 6 14:11:14 2022 -0700

    Unit test for GemNet-OC (#393)
    
    * Unit test for GemNet-OC
    
    * Load weights on cpu
    
    Former-commit-id: 28dd8fad600cd322beedf6a2d9a0ef2e52b02693

[33mcommit a00dc74b93e09d6de8defe985e6a32893a49b16c[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Tue Sep 6 14:49:09 2022 -0400

    Added more legible registry import errors (#398)
    
    
    
    Former-commit-id: e71cbb38a9307d10c6db78c32d62096f9709ed11

[33mcommit fe8427031d6ffa3203108390bab7dfe0a77dacce[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Thu Sep 1 20:41:53 2022 -0400

    Support for absolute import paths for registry (#315)
    
    * Added ability to use absolute import paths for registry
    Added `skip_experimental_import` config option to disable auto importing all python modules in the experimental directory
    
    * Removed leftover code in registry.py (which checks for "absolute_registry" registry data)
    
    * Updated `setup_imports` function to take config dict as input
    
    Co-authored-by: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 008c3f8c457a3ad3f2ca3fd0888aca189b6430fd

[33mcommit 5d52d6c6f716450487f8bb6cbf55ef6adeb33a50[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Aug 31 21:10:48 2022 -0700

    Bugfix: use correct offset distances in DimeNet, DimeNet++ (#391)
    
    * Bugfix: use correct offset distances in DimeNet, DimeNet++
    
    * Return offset distances from BaseModel.generate_graph
    
    Former-commit-id: b4d853972c7eb2acc83e2749773f7a027d11f9bb

[33mcommit 7420dbf81bff97ff9846f05cde2b8dbb67b9b7c4[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 25 14:19:27 2022 -0700

    Load scaling factors from checkpoint if available (#388)
    
    * Load scaling factors from checkpoint if available
    
    * Support both OCPDataParallel and DistributedDataParallel
    
    * Suppressing a torch.div warning
    
    Former-commit-id: 03581913adb81ec0e8efe3c9f1ace0f8efa0b5b5

[33mcommit 40b99aa2e9204a9c191df7f3d4e0d6f8b770ae4c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Aug 22 20:08:31 2022 -0700

    Bugfix: use correct otf_graph, cutoff, max_neighbors for GemNet-OC graphs (#387)
    
    * Bugfix: use correct otf_graph, cutoff, max_neighbors for GemNet-OC graphs
    
    * Add `use_pbc` to `generate_graph` as well
    
    Former-commit-id: 72d6e2aacd36bd60532a668453d656f46e55aac9

[33mcommit d8452281cba4731a44edcc5cf8aca9982cffb8c2[m
Author: jmusiel <73840022+jmusiel@users.noreply.github.com>
Date:   Wed Aug 17 18:45:13 2022 -0400

    Added scale file loading as dictionary in place of a filepath (#386)
    
    
    
    Former-commit-id: f669be23fd2d4956a20f4d53123e8f54cec05182

[33mcommit 8175416981cedef6ac2ef05a0135e7b3ce94d319[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Aug 17 13:44:31 2022 -0700

    Return tags in `AtomsToGraphs` since GemNet models need it for inference (#385)
    
    
    
    Former-commit-id: e6fddb1c10a1c3f05e4eeaf149e71b169406d079

[33mcommit 13960bf5fffa8fb2ff432ac1d230c91c9266c45b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Aug 14 17:49:21 2022 -0700

    Bugfix: Make `OCPCalculator` support both list and dict dataset configs (Refs #382)
    
    
    
    Former-commit-id: b93a5927c42a25025bbc51f2bb7bd9d19f4dea0e

[33mcommit c21d9d7d7492c5ae3c69f993742a3e3eb46f9967[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Aug 5 17:24:17 2022 -0700

    Bugfix: restore original params when predict is called with `per_image == False` (#381)
    
    
    
    Former-commit-id: 51efe6d58116a12b688ff4d2b3613f8813e36d4c

[33mcommit 929bb849c81b70101f742035909623880f258d32[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Aug 5 16:44:34 2022 -0700

    Throw error if scale file is missing for GemNet-OC (#380)
    
    * Throw error if scale file is missing for GemNet-OC
    
    * Explicitly point to scale file in the GemNet-OC readme
    
    * Minor
    
    * Update MODELS.md with config and scale paths
    
    Former-commit-id: 0b53ac496c8c80535fb95f775335b4e9ae47bbec

[33mcommit 5d969cf4b3bec303c1269a0001a01f50250c57de[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Thu Aug 4 13:47:50 2022 -0400

    Consistent graph generation implementation across models (#378)
    
    * adds a max num element option
    
    * implements non pbc cases for painn
    
    * updates spinconv pbc implementation
    
    * updated base model to calculate graph properties
    
    * updates all models with the base model code
    
    * updates all models with the base model code
    
    * bugfix: don't normalize edge_vec in base
    
    * restructure graph gen logic to include all cases and warnings
    
    * look for offsets only when use_pbc is True
    
    * updates forcenet
    
    * fix forcenet by adding use_pbc flag
    
    * changes method name to generate_graph
    
    * change variable name to distance_vec
    
    Former-commit-id: f6c96fe563e0734a5aac6b889fc02401619202e7

[33mcommit 27fe6d631b293ade5e703a10da49f0d3d15602eb[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Mon Aug 1 16:54:39 2022 -0400

    Adds otf_graph flag for GemNet-OC (#377)
    
    
    
    Former-commit-id: e227cd00e55093d12e296bc06facd1cf03741530

[33mcommit a30b8977dc9a060a6282931bd33c7f8f741fe4f1[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Mon Jul 25 18:46:09 2022 -0400

    run conda install and create env together for resolving errors when no cache (#374)
    
    
    
    Former-commit-id: f6e9992350201ce2c97a141364670cb6671378f7

[33mcommit 50c924a9eeb99d75f62dc3b533ef476552795705[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Mon Jul 25 16:09:02 2022 -0400

    Fixes PaiNN's stability issues (#373)
    
    * avoids dividing by zero for stability
    
    * Updated config.yml
    
    * Updated config.yml
    
    * Updated config.yml
    
    * Updated config.yml
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 786ef114adf94c448e796a6212d84753146df2dc

[33mcommit 12a4dd40ee2ce449864ec625b22c530dfd2a1d27[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jul 22 16:09:08 2022 -0700

    Log correct job id for slurm array jobs (#372)
    
    
    
    Former-commit-id: ae138151ff9f05aea131cd917e3d9534aefcc407

[33mcommit 9637b5c4f88b98b7a44466c8e8e6a37863019b43[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Thu Jul 21 18:16:53 2022 -0400

    Covers all checkpoint loading mismatch possibilities (#371)
    
    * cover exhaustive ckpt loading possibilities
    
    * concise logic
    
    * more concise logic
    
    * added comments
    
    * better readability
    
    * added strict_load info in example config
    
    Former-commit-id: 9b3d047e6db1731b930863d5df88308e067d5bd9

[33mcommit c7a10d1ab88480d03ceb6a1387fe17f227aa0844[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jul 20 15:43:21 2022 -0700

    Relay keyword arguments through from `OCPDataParallel` to the model (#370)
    
    
    
    Former-commit-id: 4e6512ffa189949517c47e0a44585366f38adc0d

[33mcommit f1c77f61d2035893164ad19d4a6ab9d71958c234[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Jul 15 12:20:30 2022 -0700

    Implementation of graph parallel training (#364)
    
    
    
    Former-commit-id: 01eb82abab2579b8de13fe47900df2af343c396b

[33mcommit abc3728ceeee6a9421c26829f2b7986199a5f5c3[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jul 15 06:59:25 2022 -0700

    GemNet-OC readme with test metrics (#369)
    
    
    
    Former-commit-id: f6610eba5d886fd89ab998ce2f10f1c31e8d94fe

[33mcommit b61ec3b72911317b035729e15fedf459fbf7b9bf[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jul 13 13:47:06 2022 -0700

    Catch missing edge_index TypeError in data_list_collater (#368)
    
    (+ suppress a couple of torch.div warnings)
    
    Former-commit-id: da2a23631d2527c47c0ffa74598815d365368527

[33mcommit b892033de817a07ed71a4da688e9ee003b24c134[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jul 12 13:01:52 2022 -0400

    Bugfix: Data created with PyG>2.0 (#367)
    
    
    
    Former-commit-id: 8ec6cff61604ae3c8a4a85caa0d8b432df6d7edd

[33mcommit 2e02e36c4f1bd4e2de1e4895ce253388b6734f18[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jul 11 12:51:26 2022 -0700

    Remove `is_vis` flag from fit_scaling (#365)
    
    
    
    Former-commit-id: 0a325e286796256721b08983380b72be483f9028

[33mcommit a2f07eb852d2a98a7bb7c4213b13461de0cd78ef[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Jul 10 15:07:24 2022 -0700

    Spherical Channel Network implementation (#362)
    
    * Added the SCN model
    
    Former-commit-id: 5bea37fae23d37712fa3ba59331586a925fda3a2

[33mcommit 997b52f69d39b396ce23b0bbf4317382f2c9b074[m
Author: Johannes Gasteiger, né Klicpera <gasteiger@mailbox.org>
Date:   Thu Jul 7 21:47:55 2022 +0200

    GemNet-OC model implementation (#363)
    
    * GemNet-OC model implementation
    
    * Factor out interaction indices, register GemNet-OC
    
    * gemnet-oc configs]
    
    * Adds download links to GemNet-OC pretrained model weights
    
    * Typo
    
    * remove unused config arguments
    
    * update ci resource
    
    * revert ci resource
    
    * ci debug
    
    * ci debug
    
    * ci debug
    
    * ci debug
    
    * cache test - trigger build
    
    * reorder ci setup
    
    * add license header
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 935ea21b4d0afc27815ff60a832bbfaf3ca71cc8

[33mcommit 3af283f4556b1b08965c06381ddeb48a10d5ecba[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jun 24 22:46:47 2022 -0400

    Additional OC22 configs + example configs (#359)
    
    * oc22 configs
    
    * update example scripts
    
    * OC22 PaiNN configs
    
    * `painn_dev` --> `painn`
    
    * Updates docs in example yamls
    
    Co-authored-by: Brandon Wood <bmwood@fb.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: fee33ee1b716c95585a9966bd2b9aa08499194c7

[33mcommit dddcbc5b71b33b1abaf6b94a8889713e742c4654[m
Author: Lin <879414458@qq.com>
Date:   Tue Jun 21 16:42:13 2022 -0500

    Correct OC22 download address (#361)
    
    Remove an unexpected character `%7C` in the OC22-S2EF download address
    
    Former-commit-id: 3db67bcfe3caf4e4d32effc64f2640a9bf726d55

[33mcommit 37f12fa71a49c117a0bf68d3e7c8b6620a2cc77e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Jun 20 20:46:20 2022 -0400

    OC20 Bader charge data release (#360)
    
    * Bader charge data downloadable
    
    * CI trigger
    
    Former-commit-id: cd1145c51981dc5867b680a0b1788b73754374e8

[33mcommit d7837d850a30abe0065a57c0a4e65d54f9102f93[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Jun 20 14:46:31 2022 -0400

    OC22 Dataset release (#358)
    
    * oc22 dataloader + base configs
    
    * base joint configs
    
    * push linref coefficients
    
    * add spinconv s2ef configs
    
    * update configs
    
    * gemnet-dt configs
    
    * bump ci mem
    
    * update readme
    
    * OC22 dataset links
    
    * rename oc22 lmdb
    
    * resolve broken links
    
    * fix oc22 import
    
    * remove old comments
    
    Co-authored-by: Janice Lan <janlan@fb.com>
    Former-commit-id: b2dee95a6246574be6402cb0b90d3e67c4a929fd

[33mcommit 240616b89dae5a20a2da86a3ad644bbd16b8e23c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jun 10 16:22:42 2022 -0700

    Make sure pos_relaxed and y_relaxed are not None in val run_relaxations (#356)
    
    * Make sure pos_relaxed and y_relaxed are not None in val run_relaxations
    
    * Backward compatibility with PyG 1.7.x
    
    Former-commit-id: 2299abee85ead36d0ab4d154ef95a0bf0d0b6d94

[33mcommit 7e207b5aef9ba80b4d4fec440a9e499055d2b028[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Jun 5 15:54:23 2022 +0000

    fixes energy trainer bug in #349
    
    
    Former-commit-id: 6378decc037bbef4627311147466d69dc9386370

[33mcommit 8fa6ccee3ebf06a858529a6fec1ae96abe8c8c9d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jun 3 18:24:40 2022 -0700

    Fixes bug introduced in #349
    
    
    
    Former-commit-id: 01a2364962f60737eb1a670468e8062a47df238f

[33mcommit d599a73cb9bc965cea33f93bf204d45ce5f3c154[m
Author: aarongarrison <84917438+aarongarrison@users.noreply.github.com>
Date:   Fri Jun 3 15:59:59 2022 -0400

    Fixing cell_offsets error in DimeNet++ for use_pbc=False and otf_graph=False (#350)
    
    * Fixing cell_offsets error in DimeNet++
    
    * Formatting updates
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 18800299b2aa5c7f7b146db1f4863b146f550900

[33mcommit 46a35357fcb3fafccf510de2498f59395e3f9dcd[m
Author: jmusiel <73840022+jmusiel@users.noreply.github.com>
Date:   Fri Jun 3 15:47:53 2022 -0400

    fix to gemnet scaling.py to detect if file does not exist and proceed but log warning if it does not (#340)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 2a0e97fa3b1b80317d11f290da159757604a7bd8

[33mcommit ed0b11f2db240bf05adf4baf577e37c08f2dcbf8[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jun 3 11:08:17 2022 -0400

    Update environment installation (#352)
    
    * update pyg wheels
    
    * pyg from source
    
    * revert pyg source commit
    
    * black update
    
    * update black
    
    * stable PyG version
    
    * Bump up the available memory
    
    * Reverting back to pytorch 1.10, python 3.9, cuda 11.1
    
    * Full circle
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 2be48513de9bf7435b973a9acff2d3605cf961ff

[33mcommit bb2301bc6634042cccaf16034766fdc6423a5b47[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jun 1 17:37:20 2022 -0400

    bugfix PR#343 (#351)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: a80c68520982feeb4060f7a87f7f6debf46843e8

[33mcommit 041596977f7f0038e77f78ead7dcd7eb1be125ea[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue May 31 18:25:28 2022 -0700

    Make sure best val metric is loaded when restoring checkpoint (#349)
    
    
    
    Former-commit-id: 5fb90e7e68d411299a6a33a50d1f50f103ac7390

[33mcommit 2e93f961816f037322392bf4996ed9cfd799a1ed[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu May 26 13:34:03 2022 -0700

    Reorganizes readme files (#347)
    
    * Reorganizes readme files a bit
    
    * Switch from video to gif
    
    * Minor
    
    Former-commit-id: 63273fbc19b126cc6db34edbed8ce1169f7f4422

[33mcommit 4ef1943f3da9dca38d47e74d415668398d4b2d5c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu May 26 11:08:11 2022 -0700

    PaiNN implementation, configs, checkpoints (#344)
    
    
    
    Former-commit-id: 70a68630ccf07bccf5887939c8fdfa17a38f500d

[33mcommit e339b6f9b479377a17aac571de367fc7268017c3[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu May 26 13:56:20 2022 -0400

    atomwise l2 bugfix for non-ddp  (#346)
    
    * atomwise l2 bugfix for no-ddp
    
    * all losses through ddp
    
    Former-commit-id: 26611ed349092881482027d79b1b22a3d0901f0a

[33mcommit 206934f0fb28303cb50f33f31832af57293d6e71[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu May 26 08:09:07 2022 -0700

    Bugfix in val best metric check for checkpointing (#345)
    
    
    
    Former-commit-id: 51c0755bea38f306aae25791efaca2373ee0fb74

[33mcommit c7c980615782222988a7961ea91a7c4f4834f03e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri May 20 19:44:38 2022 -0400

    Atomwise L2 loss for forces (#343)
    
    * Corrects relative weighting of force loss with no. of atoms per structure
    
    * Infer batch size from sampled batch instead of config
    
    * Set default force coefficient = 1 for AtomwiseMSELoss
    
    * Atom-normalized loss works out with the sqrt
    
    * Missed a few changes
    
    * Cleaning up
    
    * Example configs for IS2RE and S2EF
    
    Former-commit-id: 2fc58ab1c826b321f2dd9b7473ad491915b34f58

[33mcommit 4edfbe2851bae983e7f3cca60144d453c288ae8d[m
Author: Brandon Wood <bmwood@fb.com>
Date:   Wed May 18 17:50:01 2022 -0700

    Remove extra context on GPU0 for distributed training (#342)
    
    
    
    Former-commit-id: 60c327315b7cb9e7292c5f55484c1b7a233dafaa

[33mcommit a9c842cb239ac5e9197b13ece6142e60e710567d[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Apr 27 14:52:51 2022 -0500

    gemnet scaling script fix (#339)
    
    
    
    Former-commit-id: 3c6ab160843e9575b3b6c03d828965598e4f1390

[33mcommit cff21c1178d990e4ccd12d78c26b700b94c4038d[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Apr 12 10:33:51 2022 -0400

    OCPCalculator CUDA support (#333)
    
    
    
    Former-commit-id: c1c2f761bd8f07d7d5160abbef83b94c2d44121f

[33mcommit d2aaaeb67687785a99f869b4568e59cbbfab7acb[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Apr 1 21:18:33 2022 -0700

    Added an option to disable DDP for relaxations (#312)
    
    
    
    Former-commit-id: 2bdd31435b29d7f84324d57d4d6a4fb67f68c57b

[33mcommit 0c097b80b8704d4c4958fc9f6382b88ce83b2cdd[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Apr 1 16:13:44 2022 -0400

    bugfix for dict dataset definition (#329)
    
    
    
    Former-commit-id: 421816c6ec8a1c3e1cd737d6bcb99d527f253536

[33mcommit 535efb5e45d7386b98ce617bd73674a9aeec20d8[m
Author: Nima Shoghi <nimashoghi@gmail.com>
Date:   Wed Feb 23 09:49:08 2022 -0800

    Update `conditional_grad` decorator to use `functools.wraps` on the original function (#326)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 0686efe52bb4ec9b2a402bc2d82cf54490f334da

[33mcommit c76f4265577c6b7d2966f3c8867d969851d4d4de[m
Author: Nima Shoghi <nimashoghi@gmail.com>
Date:   Tue Feb 22 17:02:01 2022 -0800

    Bugfix in PyG 2 support for data objects (#327)
    
    * PyG 2 Support Regression
    
    * Fixed error for trajectory dataset
    
    Former-commit-id: 6027f0e7cde94ff0ad05db951c65f30647159a4b

[33mcommit 6e0a515d332046636b2cb653025fb458b7c15b02[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Mon Feb 14 11:52:22 2022 -0800

    Log unused parameter names on DDP errors (#317)
    
    * Log unused parameter names on DDP errors
    
    * Uncomment the checkpoint assert
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 59a6697173f674f99ef4a17ee00f4cd023467e88

[33mcommit 2745761db9d7dda4f78678217f6e59d2a62785c5[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Feb 14 14:10:06 2022 -0500

    Updates `RandomRotate` to return correct rotation matrices following PyG bugfix (#325)
    
    
    
    Former-commit-id: 397c57da217c1e6543e6aeba0b4acbcb1fb6352c

[33mcommit 3fb8f795a562de92c2cc1203e573a6527fd01aec[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Feb 14 10:14:04 2022 -0500

    add universal dataset reader (#310)
    
    * add universal dataset reader
    
    * backwards comp
    
    * remove old datasts, add flexibility for custom datasets, etc.
    
    * fix imports
    
    * rename
    
    * backwards compatability
    
    * cleanup
    
    * move datasets to base
    
    * PyG 2.0 support
    
    * rename dataset, remove is-vis
    
    * rename registry
    
    Former-commit-id: b6d42d79cf32f6e1b5508e616696163843ad41a8

[33mcommit 650a4c050eb3f43d3c20704db2a1b620ab198eaa[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Feb 10 15:48:15 2022 -0500

    decorator direct force bugfix (#323)
    
    
    
    Former-commit-id: 39f7a4d8d21ae9ee46a6fc5496cb8c558c51c6f0

[33mcommit 6564c8e5973f3c598626f3f7e637b44fae92edbe[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Feb 2 17:16:59 2022 -0500

    PyTorch 1.9.0 (#322)
    
    * pytorch/pyg 1.9 update
    
    * pyg 1.7.2
    
    Former-commit-id: 10d747430d5052cdd4b362ba29f27dfc3965f313

[33mcommit 386ee3c1afe5ce94547ce386c8eb08aedcda546b[m
Author: Nima Shoghi <nimashoghi@fb.com>
Date:   Tue Feb 1 10:07:49 2022 -0800

    Added support for PyTorch Geometric 2.0's updated Data format (#316)
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 4698882cd43df725bafb2c78ca94a1a9124fdc0f

[33mcommit 5bfe8a035d240839a62ec8a0d412096448482707[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Feb 1 12:52:31 2022 -0500

    demjson env fix (#321)
    
    * env fix, remove demjson
    
    * remove demjson import
    
    Former-commit-id: 352f1876fff42e9ef0b642d08144376e6af83d3b

[33mcommit b7024b72353062e8a73f78f9fd93a15a232bf437[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jan 21 16:48:43 2022 -0500

    remove ipynb from language stats (#318)
    
    
    
    Former-commit-id: a6ac8ee62a794c2258d5a206040e85381c11fbae

[33mcommit d8b9ead694d4632e183641c22700a893477bdea3[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jan 11 15:47:00 2022 -0500

    CCAI OCP Tutorial (#314)
    
    * organize tutorials
    
    * tutorial doc
    
    * update tutorial link
    
    * Update README.md
    
    * add colab notebook
    
    * Update README.md
    
    * created using Colaboratory
    
    * remove old colab notebook
    
    * Update README.md
    
    * typo
    
    * link colab directly
    
    * suppress large outputs
    
    Former-commit-id: 40273e27896613afa2d1f4c6b5488eb4401a3414

[33mcommit 22372b5a056d7c84641f77bd40498b01d87afe24[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Dec 17 13:46:12 2021 -0600

    no-pbc gemnet support (#311)
    
    * no-pbc gemnet support
    
    * max-num-neigh to radius-graph
    
    Former-commit-id: 8db9e1f52631a344a63c89ba2c2032a2a08d45c8

[33mcommit 385ffce5a2b019ed3de6e72f0b36d46261862b63[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Dec 13 21:25:44 2021 -0800

    Speedup relaxations and support for resuming relaxations (#309)
    
    * Fixed empty_cache to speedup relaxations and added support to resume relaxations
    
    Former-commit-id: 03ec1975be0f02a585a712b17e1036891c9d37e9

[33mcommit 18c34743208a4c68f466d3fd6b89a6bfb43a5925[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Dec 6 09:13:08 2021 -0500

    Adds SpinConv, GemNet-dT checkpoints trained on S2EF-2M (#308)
    
    * Adds SpinConv and GemNet-dT checkpoints trained on S2EF-2M
    
    * Fixes typo
    
    Former-commit-id: 993fe45a7ac9fc8a060e3c4a940b0b2b5d7233a2

[33mcommit e508a674fb1a9af459bdbf458415d794226075d3[m
Author: Nima Shoghi <nimashoghi@gmail.com>
Date:   Wed Dec 1 13:31:31 2021 -0500

    Fixed logger_project config not working (#301)
    
    * Fixed logger_project config not working
    
    * Set the WANDB_PROJECT environment for the logger_project setting
    
    * Added support for "logger" dict config
    Reverted patching WANDB_PROJECT env.
    
    * Removed logger_project config arg
    
    * Fixed project name for cases where logger is a string
    
    * Fixed error with "logs_dir" config
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: d3bd7540d347d9bd9cb67588785a322bc203d1d5

[33mcommit ab2229a37c2ec5ea8782b06643552e1d7c0ab35c[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Sun Nov 21 21:28:57 2021 -0500

    hotfix: turn otf_graph True for OCPCalc (#306)
    
    
    
    Former-commit-id: 4a02972bed48e54d3c6b310ee3f4495535dcd1bd

[33mcommit b97c0b0d17bbed3c471aadf7ec7f1fcab345b9a1[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Tue Nov 16 05:26:21 2021 -0500

    OCPCalc Update: uses otf_graph by default and takes in dictionary as well (#304)
    
    * changed otf_graph to True and takes in config as a dictionary
    
    * turn r_edges False
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 7e06b989b4f83bc884794c525f660b89f058472a

[33mcommit 1c27cae5c091f66dfd67594fd92b8bb3a2bdd466[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Nov 15 18:48:22 2021 -0500

    swap checksums (#303)
    
    
    
    Former-commit-id: 508173fed8da56229cba27ab24d4250bf1f6f284

[33mcommit 3591b60c51d86eb1c689158ee4eecd5402806ba2[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Nov 15 18:37:55 2021 -0500

    change circle resource class (#305)
    
    
    
    Former-commit-id: a7b099bedc3b214175c5a43446746d10e99e4172

[33mcommit 3959b66427605626695c96dbbd9006da47f4ce69[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 21 18:11:00 2021 -0400

    Evaluates relaxations and IS2R* metrics during S2EF training (#299)
    
    
    
    Former-commit-id: d0abd28d283dff98163a17143a95f730a6236c70

[33mcommit 610bdf9faa71426d9aed367722bb0f16293feac6[m
Author: EricMusa <ericmusa@umich.edu>
Date:   Tue Oct 19 12:34:03 2021 -0400

    Added modified K-hot elemental embeddings from QMOF CGCNN repo (QMOF_KHOT_EMBEDDINGS) (#296)
    
    * added K-hot embeddings from QMOF repo, modified from the original CGCNN elemental embeddings
    
    * modified CGCNN constructor to accept variable-length KHOT embeddings and fixed QMOF_KHOT_EMBEDDINGS imports
    
    * replaced embedding objects with str placeholders in CGCNN model construtor
    
    Former-commit-id: 0cd32a789fa7d23225c97f31efe580bc945618e0

[33mcommit 766492125961d9b7a87ad6afefe43700b052c375[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Tue Oct 19 09:46:44 2021 -0400

    Updated OCPCalculator to not require config and data paths (#297)
    
    * updated OCPCal to not require config and data paths
    
    * minor reorg to fix the config_yml case
    
    * minor fix for cpu only checkpoint loading
    
    * includes base yml and fixes it path
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 0bab6c85c58e6c115772d91d4576061195bfc98b

[33mcommit 3b9edb4c586674d608811585545abd1c4b1ed057[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Mon Oct 18 11:13:55 2021 -0400

    Loads checkpoint cleanly for non DDP checkpoint and parallelized inference (#295)
    
    * load checkpoint cleanly for non ddp run and parallelized inference
    
    * adds the missing load line
    
    Former-commit-id: 517ae4b17b5476045d401291623c4323fe56c744

[33mcommit 05d69910dbb0fd0cf6e3b267a3914daca0022a3a[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Mon Oct 18 11:04:53 2021 -0400

    changed default trainer to energy (#298)
    
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: ea965a43a4c50ab6e1cd3f7296e5151e64ab81ee

[33mcommit 0c6e4f37bfd9ca5e1752619424f1c6d5eb107cb5[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Oct 6 14:10:02 2021 -0400

    add data-path arg to download script (#294)
    
    * add data path arg
    
    * Update README.md
    
    Former-commit-id: 35ed17a9e015a7046934d3c471024b99a6d42105

[33mcommit 0b091d9a3f72a996fe557360b16d15f776c4ceca[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 30 19:21:22 2021 -0400

    remove obsolete checkpoint (#291)
    
    
    
    Former-commit-id: 7baa622d8f0a3327cab74f8ef0fa52c7363d1033

[33mcommit 59465a483f5200027dda4892b895b0998ea5a21b[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 30 16:10:09 2021 -0400

    bugfix loading best_checkpoint.pt (#289)
    
    
    
    Former-commit-id: 406569dc5e5b14d232718845adcabb0e1f22cb3d

[33mcommit b9d1b6d1285d1d8ca60f559898138376fceed739[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Sep 29 11:06:15 2021 -0400

    update oc20 data mapping (#288)
    
    
    
    Former-commit-id: 24b1579885d62a17613c0f2090c4deeb7db077d8

[33mcommit b9095b33a9bdec0f7ab205553a6ddfdc2894c297[m
Author: Johannes Klicpera <j.klicpera@mailbox.org>
Date:   Sun Sep 26 02:04:56 2021 +0200

    Fix shared parameter gradient scaling (#283)
    
    * Fix shared parameter grad scaling
    
    * Fix shared_parameters in GemNet
    
    * Warn once if a shared parameter has no gradient
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 60bcc6178ea31c183f8533acc34919866a7511c7

[33mcommit 2ad7c334aa62e482a286694a0d6e36c5501944a1[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 23 16:40:09 2021 -0400

    missing device definition (#285)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 8ed0e5704cb1fb441767f2a7919c845291cda6de

[33mcommit 377172fc14ee523627171b2d4b8859c67d7819af[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Sep 21 17:18:56 2021 -0400

    explicit cpu map (#282)
    
    
    
    Former-commit-id: d870cc1a31b9a662e01f978d36c435054713449e

[33mcommit 7d67c2b45ca32e6dd4b818c91653cd198d741191[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Sep 20 09:00:05 2021 -0400

    Adds download link to the challenge dataset (#281)
    
    * Adds download link to test-challenge data
    
    * Link to OCP challenge webpage
    
    Former-commit-id: b91b6e5468eeeebed8475c3b804230524d048e9e

[33mcommit 62218c6d945995e17183fbf4af6f1eb438f70f44[m
Author: Johannes Klicpera <j.klicpera@mailbox.org>
Date:   Sat Sep 18 04:50:35 2021 +0200

    Correctly calculate distributed loss average (#269)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 5d31cc1161a020db3b4d2d8a448b0441b97365a7

[33mcommit 847b623d98279bd4681c8589547073982c95777d[m
Author: Johannes Klicpera <j.klicpera@mailbox.org>
Date:   Sat Sep 18 04:30:26 2021 +0200

    Fix radius_graph_pbc's 1 unit cell assumption (#268)
    
    * Fix radius_graph_pbc for >1 unit cell cutoffs
    
    * Automatically obtain device in radius_graph_pbc
    
    * Split out get_max_neighbors_mask
    
    * Use faster segment_coo instead of bincount
    
    * Don't return None in get_max_neighbors_mask
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: eb7ee1b564bed808763b25a2c604ad3b503be480

[33mcommit 351153b46d81d57664796582683329b6c887b025[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 16 16:30:32 2021 -0500

    challenge submission script (#279)
    
    * challenge submission script
    
    * Fixes typo
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: e44384756e9e61d639c6ae27928d7d0eacca573a

[33mcommit da075695be3e750e6deb2898e8ac24420ed93812[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 16 14:26:53 2021 -0500

    add missing spinconv import (#278)
    
    
    
    Former-commit-id: b19dc93b1b724359de0fe7f436b3b94fcfa80988

[33mcommit 1aa28863ecb5e0c45862c09a353e14bd865c59c3[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Sep 15 15:57:07 2021 -0500

    Script to generate metadata for load balancing (#277)
    
    * metadata gen script
    
    * black
    
    * add load balancing to readme
    
    Former-commit-id: 2322807cf608f6cf5d83334059a6897487b1e100

[33mcommit 8c58db1230c1a521a918d49d3cac621024e4e09f[m
Author: Johannes Klicpera <j.klicpera@mailbox.org>
Date:   Tue Sep 14 21:21:23 2021 +0200

    Load balancing batches across GPUs (#267)
    
    * Remove args.num_workers
    
    This is a duplicate of config["optim"]["num_workers"]
    
    * Implement BalancedBatchSampler for load balancing
    
    * Fix relaxation trainer
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 20c72dcb6755a080c468f1f9e540b9e6099fd258

[33mcommit b67dced442016797367805c5dca06dbe15b4b3ff[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Tue Sep 14 11:50:43 2021 -0700

    Makes Ray Tune optional and fixes submitit checkpointing bug (#272)
    
    * makes ray tuneoptional
    
    * Ignore flake8 errors
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: ae6b9b9c85e5207475f3bcd4214157a099b46812

[33mcommit 6ae621b8cc39ec9391dd91cd5fb825f93e8c7911[m
Author: Johannes Klicpera <j.klicpera@mailbox.org>
Date:   Tue Sep 14 20:11:58 2021 +0200

    Fix DimeNet triplet indices (#270)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 8d156f03386164dc01be1e33611d3ae1e21ffe8b

[33mcommit 6cbc0de3178562bdb984fdd33e2cdcbf0efcb525[m
Author: Johannes Klicpera <j.klicpera@mailbox.org>
Date:   Mon Sep 13 16:45:42 2021 +0200

    Reduce test_cosine_similarity check to 2 decimals (#276)
    
    
    
    Former-commit-id: 94719bea9443d0d3d3023290cc6f4e4b2074388f

[33mcommit 150ea186f02d09222e5a3842096d737a1128f2a4[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Fri Sep 10 16:39:13 2021 -0700

    update model name in gemnet fit_scaling (#274)
    
    
    
    Former-commit-id: d508b560272c1f328a45ee7a5fe1e6f633107c04

[33mcommit a171dae2e926ccc8d2b83d7f4bf0340622d12437[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Sat Sep 4 22:07:33 2021 -0400

    Updates S2EF notebook for v0.0.3 (#265)
    
    
    
    Former-commit-id: a3961b7d1a22ffaf3b10236fe91ee3d3f2386474

[33mcommit 2f91e22a7fe650a6bcb5cd1dcea660ca68378ef0[m
Author: Adeesh Kolluru <43401571+AdeeshKolluru@users.noreply.github.com>
Date:   Thu Sep 2 15:58:47 2021 -0400

    Added script to generate gif from traj (#259)
    
    * added script to make gifs
    
    * black reformatting
    
    * parsed the arguments
    
    * Explicitly pass povray settings
    
    * remove deprecated part and change directory before running povray
    
    * added a note and changed the requirements
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: aa44fe5ade14fe7b45ce45b64644909eb992e623

[33mcommit dc3c5196c5294725afdb23896ee2c6371c199c48[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Sep 2 14:20:27 2021 -0400

    setup v3 (#264)
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 009c422058dd70dec51098f4119721f9552cde25

[33mcommit f235b3a29b1ac84edee45ddf65b098a94b5fdad7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Sep 1 17:54:20 2021 -0400

    Import logger when setting up imports; fixes #262 (#263)
    
    
    
    Former-commit-id: 3ea859a93251faee3a68e30ee8d16d615e4e074f

[33mcommit 538b3c4596bc09c6d1d452a5f1030a44f18b4e2e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Sep 1 16:40:41 2021 -0400

    bugfix: remove epoch arg from validate (#261)
    
    
    
    Former-commit-id: 7f23049c6900d5347efaac4a6f2bdb355bf0cc2e

[33mcommit 49763a60868f051c223352732602778138286534[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Aug 27 20:03:25 2021 -0400

    Support for ignoring specified directories within `experimental` (#257)
    
    
    
    Former-commit-id: 17f593c5e966eb6e20c97a51b267968a7dbe0f46

[33mcommit 30d5ddd83ff86186e3cdbc99325576cbfa0989f1[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Aug 27 14:07:00 2021 -0700

    Remove FAIR cluster paths from MODELS.md
    
    
    Former-commit-id: 711553a837c9be00be0ae9648886f7c346e52249

[33mcommit a4a242fa785098650abbda98c044281c0304519b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Aug 27 14:46:09 2021 -0400

    v0.0.3 release changes (#256)
    
    * Changes for v0.0.3
    
    * Reformatting MODELS.md
    
    * GemNet scaling factors
    
    * Env update
    
    * Correct pytorch version in readme
    
    * Update OCPCalc docs
    
    * gemnet scaling factors file
    
    * remove unused var
    
    * remove redundant lines in collate fn
    
    * Updates gitignore
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: fc646914876f8b48968e386737522fdccad5b151

[33mcommit ec90c90ccc20630712df12408dc0293a2bb088dc[m
Author: Alex Schneidman <alexmarkschneidman@gmail.com>
Date:   Tue Aug 10 09:08:03 2021 -0700

    added instructions to readme which fix bug with install on mac machines (#255)
    
    Co-authored-by: Alex Schneidman <alexschneidman@devfair0117.h2.fair>
    Former-commit-id: 19558620b327339e64b5ce58b86edefc3de837bb

[33mcommit b9abc8051b70f15e18b5737773a38c4389a26c7a[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jul 27 12:06:51 2021 -0400

    IS2RE relaxation scripts (#253)
    
    * init is2re relaxation scripts
    
    * add is2re relaxation documentation
    
    * add path to sample config in docs
    
    Former-commit-id: 209fb9f97eca720415270a34893d868963e66e54

[33mcommit 2976704988a7820874f98152bd1b6556ec942650[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 19 09:12:06 2021 -0700

    hotfix: tutorial link
    
    Former-commit-id: ea2238ab7ab8a576a34f4a83da319ae7f247ebb4

[33mcommit b6fe3dd133506632beca70a5c389430bff1cd6de[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Fri Jul 2 14:02:49 2021 -0500

    Update configs from epochs to steps convention  (#245)
    
    * updating is2re configs from epochs to steps
    
    * All configs converted from epoch to step LR milestones and warmup
    
    * added exception to warmup_lr_lambda to catch when LR milestones and warmups are defined in epochs
    
    * negative sign removed from eval_every
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 0ddc5164af20186545baabfcd328fdeb7f4cc7e9

[33mcommit f34def64fd006bb724772140ad4f690e9202b630[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jun 16 13:06:29 2021 -0400

    #GPUs>#LMDBs bugfix (#248)
    
    * traj-lmdb restructure
    
    * traj-lmdb sampler
    
    * deterministic shuffling ddp
    
    * readme cuda/torch version
    
    Former-commit-id: 8627c16a50d37cc20e223ed77cbb060294740152

[33mcommit 6d591e1d4564680ff3346bc105f39187ea2d628b[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Wed Jun 16 13:01:54 2021 -0400

    Enabling cpu only predictions using main.py (#250)
    
    * Accomodating use of OCPCalculator for energy calculations
    
    * fixing formatting and adding per_image arg to energy_trainer
    
    * corrections as noted
    
    * fixing energy attribute type
    
    * accomodating cpu only predictions
    
    * reformating for improvment
    
    Former-commit-id: 2c7aafa606f565316a14a706d5b260cbabe9ef2b

[33mcommit bb4cee2c0c837fc4073f32061e8401e1ffefd571[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jun 16 12:43:11 2021 -0400

    Pytorch 1.8.1, Python 3.8 Support (#247)
    
    * torch1.8, python 3.8 update
    
    * torch_geometric install
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 6481e70e9a4b7e1ee02dad72a625b36cb5f757d5

[33mcommit 8e7623ec9662ca665c44a604ff7cc24684350294[m
Author: Brook Wander <73855115+brookwander@users.noreply.github.com>
Date:   Wed Jun 16 10:30:26 2021 -0400

    Accomodating use of OCPCalculator for energy calculations (#243)
    
    * Accomodating use of OCPCalculator for energy calculations
    
    * fixing formatting and adding per_image arg to energy_trainer
    
    * corrections as noted
    
    * fixing energy attribute type
    
    Former-commit-id: 92afcc19b48d9e0d1cef8deec1511ba12c43dafb

[33mcommit 5c86206af4e9347dbb5840f7a734e5b1f9f85741[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jun 8 10:38:01 2021 -0400

    update citation (#246)
    
    * update citation
    
    * update citation
    
    * update citation
    
    * missing bracket
    
    * Indent
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 5f5ca5335530e7feec7615bc98191a224bef18a0

[33mcommit 7f30c89985425a7cfefdc019368de3152f9e6962[m
Author: ianbenlolo <ian.benlolo@mail.mcgill.ca>
Date:   Mon May 31 12:12:37 2021 -0400

    :books: Fixed broken link. (#244)
    
    
    
    Former-commit-id: dfa60d21221ca5ae0d78df32dd9a3c173d9dce33

[33mcommit 49ba0663aa8532bbc5708e216a2b702456318407[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu May 20 12:20:29 2021 -0500

    update ACS citation (#241)
    
    * update citation
    
    * Update README.md
    
    * Update DATASET.md
    
    * Update MODELS.md
    
    Former-commit-id: 0354d0919df5139d348ab8f468070a455167fcce

[33mcommit 6218df5ddf757d70d7eaf60857afbf8d425525f1[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Tue May 4 12:39:13 2021 -1000

    Enables additional arguments for slurm (#240)
    
    
    
    Former-commit-id: 52422001821b31845a02c5b6d6009bca58bf8976

[33mcommit 5e32e128adc381e581e9ba97e55fc40935d1ee30[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Tue May 4 09:33:40 2021 -1000

    Adds checkpointing functionality for Ray Tune (#239)
    
    * Adds checkpointing functionality for Ray Tune
    
    * Consolidated the hpo update into the base trainer
    
    * Minor update to hpo_update logic
    
    Former-commit-id: d3c493fe5f3433cbb7bd4e5d11d025143e6b1583

[33mcommit f3ba733096fc8cf1deb1729917d1e4fb1318203a[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Tue Apr 27 10:54:15 2021 -1000

    Fix/update to scheduler module (#234)
    
    * Add explicit update_lr_on variables to LRScheduler
    
    * Makes scheduler step every iteration by default
    
    * Handle edge case for ReduceLROnPlateau
    
    * Updates energy trainer to step LR scheduler on every iteration
    
    * rop bugfix
    
    * Changes several of `BaseTrainer`'s functions to abstract methods
    
    * update variable names in warmup_lr_lambda to reflect change from epochs to steps
    
    * Use warmup_epochs if warmup_steps is not available for older checkpoints
    
    * Minor update for when scheduler state is not saved
    
    * Set mode to training every iteration since it wasn't getting reset after eval
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 73814cb18ce52cba1eb217dd21474207825ffd5e

[33mcommit d184e1f65d8b593f661e4273429625c678648c4b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Apr 25 22:18:26 2021 -0400

    Sets the seed of each dataset instance based on process rank (#238)
    
    
    
    Former-commit-id: b148ebf096b77861a9d4461aaa89d1709667fa9d

[33mcommit a661ca4d5799e5497f78b0d90e10c0473fcf3063[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Wed Apr 14 14:27:26 2021 -1000

    Support for hyperparameter optimization with Ray Tune  (#232)
    
    * minimal support for Ray Tune with Slurm
    
    * added hpo flags to all trainers, updated comments and readme
    
    * updated permissions for slurm hpo files and minor changes to run_tune.py
    
    * minor readme and comments update
    
    * HPO and wrapper function documentation added, removed flake8 noqa: C901
    
    Former-commit-id: 996aa667c62c1c23fc4cfa9fea68b56e5ab1e6cb

[33mcommit bc69996f8cfbbad2da363a7e1a6a2e65a072ea3d[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Apr 13 12:28:47 2021 -0400

    remove duplicate optimizer (#236)
    
    
    
    Former-commit-id: e32f8fc2479e265f083a1c42332ea708187c75ac

[33mcommit 690f2dc6dee94cb4439546aa199c351251f841a3[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Sun Apr 11 21:00:14 2021 -1000

    Add null scheduler (#233)
    
    * Adds a null scheduler
    
    * add documentation to scheduler class
    
    * minor modification to documentation
    
    Former-commit-id: 2575283e2184b6dc0d3860cc6586fa2ffde7fe69

[33mcommit d767ae58146fcf3c2d26f822b040e5383c0e1638[m
Author: Sterling Baird <45469701+sgbaird@users.noreply.github.com>
Date:   Tue Apr 6 18:29:42 2021 -0600

    Update getting_started.rst (#230)
    
    link was missing a `y` in `catalyst`
    
    Former-commit-id: fd6fb3be78547b828d875b7449ba3d44214b2173

[33mcommit 4022027fa3d7636bd39cba79d5163ca3e3f604f1[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Apr 6 09:16:34 2021 -0400

    Support for multiple schedulers (#226)
    
    * flexible schedulers
    
    * energy trainer, step fix
    
    * raise error for no val set
    
    * missing catch
    
    * modify citation
    
    * load/save scheduler
    
    * add lr to log
    
    * fix - ROP scheduler doesnt support get_last_lr()
    
    * copy config
    
    Former-commit-id: 971bce9b4fcdad8e5f489431ffad17408d04a112

[33mcommit ab27600da7d00a3cdf2ab1bb93ed954e7c39d3c9[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Apr 5 16:31:38 2021 -0400

    deterministic tests, dimenetpp test (#228)
    
    
    
    Former-commit-id: 498b227e5fc2a86ada5dbc185101b8d1defc91e2

[33mcommit 40a64d62bb90d9f01d5318b633005a55ed8a1f92[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Mar 30 18:05:55 2021 -0700

    Add per adsorbate trajectories (#225)
    
    * Add per adsorbate trajectories for IS2RES
    
    * phrasing
    
    * Update DATASET.md
    
    * Add adsorbate symbol
    
    * note on adsorbate test splits
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Co-authored-by: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
    Former-commit-id: 9c51bc9c1f9120ba36ccbbe4bf41afb909949399

[33mcommit 5fd2c3a99500464bc662e4525c0e7133b0f713ad[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Mon Mar 29 13:33:06 2021 -0700

    Add rattled and MD downloads (#224)
    
    * Add rattled and MD downloads
    
    * Add missing sentence
    
    * adds rattled+md to download script
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 854e2ccbb1057d6864630ce824c9176a1f0c7c83

[33mcommit da2bc4c555098975cc2bf68afb3d8fff238ea64f[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Fri Mar 26 18:06:48 2021 -0700

    Modify data mapping pkl (#219)
    
    * Modify data mapping pkl
    
    * Address review comment
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: f69886aa947fc5dd2460908511a4de4a89a065f0

[33mcommit b8396071242853fa880c0e1dfa8c213009afaef2[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Mar 26 17:51:31 2021 -0700

    Cast `step` to be int for wandb (otherwise it crashes) (#220)
    
    
    
    Former-commit-id: 53871167980c8176e0010dce8c10074279ab5f58

[33mcommit e0f3bd3084d8dfb7e39f4e9c7579392ea34ca8da[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Mar 24 17:19:03 2021 -0400

    Optimizer flexibility  (#218)
    
    * optimizer config
    
    * explicit args
    
    Former-commit-id: ece35d501f9107df8817655a715a0053d5cb86c9

[33mcommit 9c73d27a39f9068f9707c6f330ac4678f6d705ec[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Sat Mar 6 15:04:22 2021 -0500

    Additional tutorials: preproc, lmdb creation (#211)
    
    * preproc tutorial
    
    * add link
    
    * lmdb notebook, updated preproc
    
    * interacting with lmdb
    
    * cleanup notebooks
    
    * readme pointer to tutorials
    
    * tutorial readme updates
    
    * update s2ef tutorial
    
    * remove old tutorial pointer
    
    * Fixes typo
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: c71d4bbd23af1439a397fff2046a45b27e415fd6

[33mcommit 045a9e3a779aeb8bc3d5619de133e8feb2a4db8a[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Mar 5 22:33:28 2021 -0500

    Minor reorganization and updates in readme (#212)
    
    * Upgrades to pytorch 1.7.1, pytorch-geometric 1.6.3
    
    * Updates ase and pymatgen as well
    
    * Minor updates to readme files for clarity; adds circleci badge
    
    * Fixes typo
    
    * Move cite to end of page in dataset md
    
    * Readme update
    
    * Readme update
    
    Former-commit-id: aa077ad811b12286de2b7458f0069c73e72831de

[33mcommit b82086c6c6ea52d3fccb22f8664af28e93a7bda2[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Mar 5 19:16:54 2021 -0500

    Upgrades to pytorch 1.7.1, pytorch-geometric 1.6.3 (#209)
    
    * Upgrades to pytorch 1.7.1, pytorch-geometric 1.6.3
    
    * Updates ase and pymatgen as well
    
    * Fixes typo
    
    * Updates circleci cache version
    
    * circle ci cache version
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 6b31d3a341848ef563bc552c5b4f11c3ad948140

[33mcommit 15d8782f493156887f017ce52c6397a6981db243[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Mar 5 01:57:57 2021 -0500

    Remove energy prediction / target requirement from IS2RS evaluator (#210)
    
    
    
    Former-commit-id: 10de373baf1949da2417845a993b0dfe0424abf8

[33mcommit e9af14da6ec28db8d0a2578395f3b427a592d58f[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Feb 26 16:07:59 2021 -0500

    update is2re configs/cps (#203)
    
    * update is2re configs/cps
    
    * update is2re links
    
    Former-commit-id: 1ede9eebf56899dd6c962a3f06cd82c5d543632c

[33mcommit 16531322adfdd9361c5b5bc69946e6eba87a39b9[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Feb 26 12:08:12 2021 -0500

    flag to load ddp cp without ddp (#204)
    
    
    
    Former-commit-id: f0eec350a433f253db7af68e662d50c0cbf257ca

[33mcommit 6d8ae6bbef31269d3e5323527c909b2ac110bebb[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Feb 24 17:00:10 2021 -0500

    conditional decorator for energy/force trainers (#201)
    
    * conditional decorator for energy/force trainers
    
    * remove unused import
    
    Former-commit-id: f0bd931a6cd28d1d7dffef6f6e217ce28c8c145c

[33mcommit 9672b77b09327539a323a4b6cfac211a2026273c[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Feb 23 19:03:37 2021 -0500

    Resolves validation gpu memory issue (#199)
    
    * resolves val gpu memory issue
    
    * remove debug line
    
    * function annotations
    
    * removes redundant nograd call
    
    * no_grad to predict call
    
    Former-commit-id: 72a1654dcede041b68438fcb20e2a9f280d3e3ba

[33mcommit 39b5ce10ac6edf04a2e103f42945b7e4a1ed3960[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Feb 23 09:24:57 2021 -0800

    Add links to download trained DimeNet++ models (#200)
    
    * Add links to download pretrained dimenet++ models
    
    * Removes DimeNet-20M and DimeNet-All since we no longer support DimeNet
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: d92ca7ebc832277b8dfbd9cb12485f4728d6b634

[33mcommit 911d726661ee059521c7faec8e92d1d6045b9be8[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Thu Feb 11 20:02:39 2021 -0800

    Modify underlying downloads and add more download materials (#197)
    
    * Modify underlying downloads and add more download materials
    
    * Add checksum for mapping tables
    
    * download_data.py
    
    * Add changelog
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 4bfabed96a3218f9bedaf08083060cafa45ea2b3

[33mcommit 6ea02e393c6dd8b9ec6a2e2a7ad8fa02526e1374[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Feb 11 15:23:06 2021 -0500

    NPZ object fix #192 (#194)
    
    * stores npz objects as concat arrays
    
    * cumsum fix
    
    * remove superfluous call
    
    * removes last idx for np.split
    
    * idx for forces
    
    * move idxing to base
    
    * remove debug statement
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 92337ef47522e26c8a7c8e7c08fd252956ee4156

[33mcommit eda51e89d68cb077559e9265c8f5604df458c712[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Feb 10 16:51:05 2021 -0800

    Bugfix: pad dataset to ensure all dataloader processes have equal samples (#196)
    
    
    
    Former-commit-id: 4c5a35e17fb68d8b421533c13c14cc2cd0bf9acb

[33mcommit 1d32cd135a4732f7c1fba0ec673b535f4a003015[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Feb 5 16:41:31 2021 -0500

    closedb fix (#191)
    
    
    
    Former-commit-id: efaef458276133e1d4bdaa0e12b0622566273a15

[33mcommit 7745094f7e1423a6a78bda15304b5cc996d07738[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Feb 3 07:17:07 2021 -0800

    Clear cached cuda memory after each relaxation batch (#190)
    
    
    
    Former-commit-id: 1121c3fb9ca6cb9b7372e1f65003e2e588befd48

[33mcommit 8a1587524d9e3f33f333a37a1e2e27c707cc62c8[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Feb 2 14:49:05 2021 -0500

    catch for no fixed atoms (#189)
    
    * catch for no fixed atoms
    
    * reference constraint
    
    Former-commit-id: 8159bbe3217325ecfeee8fa46260e29d70e0538e

[33mcommit eeeb5885434068a98be88076a62251c6d1ff4da7[m
Author: Weihua Hu <weihua916@gmail.com>
Date:   Mon Feb 1 22:25:24 2021 -0800

    fix minor typo in README.md (#188)
    
    
    
    Former-commit-id: 7a138fa1cc6a9528f9b7b9d7e36989a8f587f2cf

[33mcommit 8735fab11211aa8b6ca187c5359177509bc22584[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Feb 1 16:29:26 2021 -0500

    pointer to notebooks, arxiv v2 (#187)
    
    * pointer to notebooks, arxiv v2
    
    * Adds DimeNet++ to readme
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 70e838295d64e4738fda6f91ecbd8bd7d9d1a9f5

[33mcommit 4ffcf84f6bc63a297837995ba34265462fb4416c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jan 29 15:26:59 2021 -0800

    Setting the same `black` version in pre-commit hooks and circleci (#185)
    
    * Setting the same `black` version in pre-commit hooks and circleci
    
    * Revert "Ran black"
    
    This reverts commit 5d94b77899f81bf8d5fee6d5be5972b931c83f3f [formerly 9b6d5f297f805bace1063c0f10454e2a3ec46b7a].
    
    * black
    
    Former-commit-id: 2ee73cfdb96ae2c6865bf66d6374db65ba137300

[33mcommit 5d94b77899f81bf8d5fee6d5be5972b931c83f3f[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Jan 29 13:43:25 2021 -0800

    Ran black
    
    
    Former-commit-id: 9b6d5f297f805bace1063c0f10454e2a3ec46b7a

[33mcommit a53ca3aba7e7a24d66179e37a0f168be33c26777[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Jan 29 13:40:48 2021 -0800

    Step-wise Scheduler (#177)
    
    * Support for both step-wise and epoch-wise scheduler
    
    Former-commit-id: 95fef675a6e8edbb179094e3360f56bda86fe9ff

[33mcommit a227207fcb614ed577f1caf14bb16df44fc46711[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jan 27 13:37:59 2021 -0800

    Adds DimeNet++ S2EF configs (#184)
    
    
    
    Former-commit-id: 0b9665b5b03efde450c1fb530cbe981d17c7959a

[33mcommit 253bd966c21aff10f8ca578928859a13185525aa[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jan 27 13:37:41 2021 -0800

    Bugfix: allow passing `traj_dir` as `None` when running relaxations (#183)
    
    
    
    Former-commit-id: e93744f4c1d7a3069e6820afe33a685368abad1c

[33mcommit 700454eee2c0a11bab5192b0b07bac6ad6e6d502[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jan 26 12:06:35 2021 -0500

    D++ IS2RE configs (#182)
    
    * dimenet++ is2re configs
    
    * remove old dpp config
    
    Former-commit-id: 644c7860486f16569becd5100588989bbc58662f

[33mcommit e948059d3a8063d8cb9461ce0226a538c57aa00d[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jan 19 11:23:54 2021 -0500

    Adds discussion board reference (#181)
    
    * adds discussion page to README
    
    * Update README.md
    
    Former-commit-id: e76c4cf5fad7a5813a1f76708728bde91c1c2bc6

[33mcommit 4cc808b8302bff025ddec4c998f359824356939f[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Sun Jan 10 20:41:27 2021 -0600

    log commit hash from outside repo (#176)
    
    * bugfix: git commit log outside repo
    
    * try/except block
    
    * set to None
    
    * try/except comment + missing describe
    
    Former-commit-id: f49727dadc0ee4c8a1e00806a81426973746d826

[33mcommit cf6d2a5c144131960e6feda7ce0a8a3521089548[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jan 8 09:59:02 2021 -0600

    logs git commit and no. of gpus used (#173)
    
    
    
    Former-commit-id: d372f6f646996c2b124ef948eb43aaa52f3cbab0

[33mcommit 14bbb33a39904e22bec5a1e459ab281b811c6b8a[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jan 8 09:58:38 2021 -0600

    clarify pretrained models (#175)
    
    
    
    Former-commit-id: 6af6520cd5ce4ecaefbc3961d542662e31e2f3d2

[33mcommit 1a7293f9de4c0aa4c92b02b5956b504e2f0e563e[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Fri Jan 8 05:00:43 2021 +0000

    trigger ci build
    
    
    Former-commit-id: 3dacd22bd186538508b17f68f49c91fdd186c98b

[33mcommit 7bed5b459b66b11aea9a7d42c86c9a7919925535[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jan 6 19:05:58 2021 -0600

    Updates demo notebook per repo changes (#170) (#172)
    
    * updates notebook per repo changes
    
    * reduce num-workers to 8
    
    Former-commit-id: d111dcd511d33fb4056e5c710f04cda7e5713ba1

[33mcommit 3a4260918c712423cdc2ddce8429b65afe031623[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Jan 7 01:39:33 2021 +0530

    Keep track of all LMDB connections within each dataset worker (refs #170) (#171)
    
    
    
    Former-commit-id: 4f25420e2d2d3638e10d77678d4723e69af213dc

[33mcommit ea20c83989ca66bbd48705db695ef2da873fadf0[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Jan 4 13:01:50 2021 -0600

    Moves ForceNet config to 200k (#169)
    
    * 200k->all
    
    * move forecenet to 200k
    
    * all->200k
    
    Former-commit-id: ddba6fd12092ce10559d78d9f8a03e6c60287f94

[33mcommit d0e9324f5b0214ebea1b9bd85b96eedeb8c6d466[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Jan 4 12:27:09 2021 -0600

    200k->all (#168)
    
    
    
    Former-commit-id: 910f9c5413330969cec895943b3ed4a2d8b340c8

[33mcommit d42e50155ea99303543272eaec8c5ca1b057de16[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Jan 4 08:24:57 2021 -0600

    update config override (#166)
    
    * update config override
    
    * remove config override
    
    Former-commit-id: d3c0d98acc31ab7ba08f04c1c9f04941cc472b86

[33mcommit 92e4f5dcb3fc3c94f32551124bf4d6411ad8e52b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Dec 24 07:25:22 2020 +0530

    Adds support for auto-requeueing slurm jobs (after timeout / preemption) (#164)
    
    * Adds support for requeueing slurm jobs (after timeout / preemption)
    
    * Make sure requeued job is initialized with original config
    
    * Renames SubmititRunner to Runner since it's used to run locally as well
    
    Former-commit-id: 7138b5ffd434559b68aecd58c344f1bd5bd26248

[33mcommit e6efacd550f7153db21ae3165c113817156cc6ea[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Dec 23 02:21:53 2020 -0800

    Running black; getting circleci to pass
    
    
    Former-commit-id: dbabf6408b46da1ecc6c6c66632372cdbc040125

[33mcommit 45fd6afeef4ff83892bedcb44881af3d134e8d27[m
Author: Joshua Hansen <j4ah4n@gmail.com>
Date:   Wed Dec 23 01:55:49 2020 -0800

    Inline concat path using os.path.join instead of wrapping in pathlib.Path (#163)
    
    This fix allows the use of alternative TensorFlow SummaryWriter's (e.g., GCP, S3, etc). Using pathlib.Path, it changes incoming --run-dir=s3://path to s3:/path (Note the single slash) and the SummaryWriter check fails to match due to the single slash.
    
    The fix is to not wrap in a pathlib.Path and do inline concatenation using os.path.join
    
    Former-commit-id: 95aec52c6f46b3011323a26277f88c4789e4ae91

[33mcommit 859ca64f4a9adaa06d5a6b384fe378a1cadb0e6e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Dec 23 05:47:09 2020 +0530

    Updates the DimeNet++ config to be self-contained (#165)
    
    
    
    Former-commit-id: a408dce586299d32141e045052f802fff365c0f5

[33mcommit e11558e8c637b51a7534903814035322d2373761[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Dec 21 13:31:39 2020 -0600

    LMDB optimizations (#154)
    
    * lmdb optimizations
    
    * close lmdb after training
    
    Former-commit-id: 3f96de861a0f37216b72a5482e92ce294bb96523

[33mcommit 14d3c7acbfbde5020c02dc4cd498799cbebe9138[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Dec 18 14:40:26 2020 -0800

    Rotates `.cell` as well in the random rotation transform (#160)
    
    * Rotates `.cell` as well in the random rotation transform
    
    * debug circle ci membory limit
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 41fb44db65d6bd75a7e0c89c36072002a990993a

[33mcommit d13bf608dfad98f12531db239c20a1bd6e95a673[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Dec 17 03:12:16 2020 -0600

    updates PyG binaries (#159)
    
    
    
    Former-commit-id: d6a54dc41cd794b3faf82a4b7462435f915ce474

[33mcommit baf10de73087b24b1dd3bd20a6df482ef9923b1b[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Mon Dec 14 14:11:36 2020 -0800

    Add data-mapping pickle and update bibtex (#158)
    
    * Add data-mapping pickle and update bibtex
    
    * Add information about download size
    
    Former-commit-id: ce89e1d875dd0dbf8899a3b864fa6e4cd4046c0b

[33mcommit 529718dde21183ca80106fa0654e28f26d1479de[m
Merge: b8c4758 c4e779e
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Fri Dec 11 16:40:10 2020 -0800

    Merge pull request #150 from Open-Catalyst-Project/forcenet
    
    Add Forcenet
    
    Former-commit-id: 64063a35f9cfebc75a7f8210359474a8fe5dedaa

[33mcommit c4e779eedb8d40fffd245486302491ddeb96a0a0[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Fri Dec 11 16:28:58 2020 -0800

    Add shape test for forcenet
    
    
    Former-commit-id: 4e6e1b4f361bc46633d17f627a83eb7f934cac75

[33mcommit fc906d8fc2a584661ad631c1fab9a07a6c78e2e3[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Fri Dec 11 16:15:42 2020 -0800

    Address review comments
    
    
    Former-commit-id: 18485cf253be08a3a3caa3e6b76ca97faafbe0c2

[33mcommit f11d810acf925aa751c5a6f7d6cffd2db0338ef3[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Fri Dec 11 12:56:41 2020 -0800

    Modify eval batch size to avoid oom
    
    
    Former-commit-id: 4c6c4a87a22e4bf683e7befb3033638d1364c8c1

[33mcommit e702be7bf263ef6181032ab6b1a0b0de89476b31[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Thu Dec 10 17:30:20 2020 -0800

    Remove debug statements
    
    
    Former-commit-id: 6ee207d5765f049da92eea377e530aab0f188e4d

[33mcommit 1a298b4be95058f6a142b162ac8e95be9701a9cc[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Thu Dec 10 16:26:56 2020 -0800

    Modify hidden channels to match default forcenet
    
    
    Former-commit-id: 4cc9dcbf10be3852bd467aeed2a5a950c9278142

[33mcommit 1f9e268c67602839443752c47d65d568500e5ca1[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Thu Dec 10 14:17:36 2020 -0800

    Address review comment
    
    
    Former-commit-id: f54f54b53065d63cdb2841ea6f21007eae9aa186

[33mcommit b8c4758da2b0be6dd3b77c3e290989ad8bb2b7f5[m
Merge: c86443b 5b54194
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Thu Dec 10 12:37:06 2020 -0800

    Merge pull request #156 from Open-Catalyst-Project/cpu_support
    
    Add cpu support for forces trainer
    
    Former-commit-id: 008037cefceb8c286d80ddcac25583e1d377a82b

[33mcommit 5b54194ad3688374cc0787e16b73ba854772093b[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Thu Dec 10 20:18:49 2020 +0000

    supports cpu energy trainer
    
    
    Former-commit-id: ee20c3a2b6c032ec40382323899ab457baf796e7

[33mcommit d8101d15eac8201da6be3c2f2e61cd35edf56058[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Wed Dec 9 18:53:34 2020 -0800

    Refactor further and address review comments
    
    
    Former-commit-id: dc6f2275971b60e117a0709af53a2519a1700fcf

[33mcommit aececec6dc1a8f1a0a834f3fa75ae4989a63da27[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Wed Dec 9 17:47:49 2020 -0800

    Add cpu support for forces trainer
    
    
    Former-commit-id: c26af224b34766413f6d5d1693ed8db01c294de2

[33mcommit 9643f66c14b5bc23cabdd7be30f9950c79cb413f[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Wed Dec 9 16:46:42 2020 -0800

    Remove old pt tensors and refactor
    
    
    Former-commit-id: 3b915e0c71b2073be269bfc93b924e7427d21e4b

[33mcommit 79d90b8af2403239d17e7f7b63f1dfe4d525b6eb[m
Merge: eb80cc6 e9b6962
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Wed Dec 9 16:20:44 2020 -0800

    Merge branch 'forcenet' of github.com:Open-Catalyst-Project/ocp into forcenet
    
    
    Former-commit-id: fc33bf7e52f1067bc2a18370115b0d3ade957bef

[33mcommit eb80cc6b7a2cf725e497ba5c3289eec8503f3215[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Wed Dec 9 16:20:37 2020 -0800

    Modify implementation for reproducability
    
    
    Former-commit-id: fce617251a7709726297a512422358d1212ba1b9

[33mcommit c86443b43a12e6c7e67c8d951aababafc3975ee9[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Dec 9 15:56:04 2020 -0800

    Correctly check val performance improvement for either MAE or EFwT metrics (#155)
    
    * Correctly check val perf improvement for either mae and efwt metrics
    
    * black
    
    Former-commit-id: 278fc88af8e2484b1663d9f07d136dc5082a6a32

[33mcommit e9b6962715c6fa8af106ec47dbb4288f7bd9802c[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Dec 9 13:33:58 2020 -0500

    includes forcenet embeddings for #150 (#151)
    
    * includes forcenet embeddings
    
    * move embedding files
    
    Former-commit-id: 5a798932f710799404d20477ca9533da8f85cd17

[33mcommit 08ec13636018ac188078a5372a8027f7495e6521[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Dec 9 12:48:29 2020 -0500

    Automate data download, preprocess, and structure (#152)
    
    * automate data download, preprocess, organization
    
    * pre-commit hooks
    
    * argparse to main script
    
    * add s2ef preprocessing flags for ease of lookup
    
    * updated download instructions
    
    Former-commit-id: fe7d01117151cd2ad12db966fea1e320d7f243ba

[33mcommit c056f9059893c63623746bd5e9510dd3a5c9f4c8[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Mon Dec 7 18:08:18 2020 -0800

    Add default params and fix sph handling
    
    
    Former-commit-id: 902c8c80bd4d650c4aa6ac5559c2ef5430450c4a

[33mcommit bc69e938f9e47bfb5036053f4d5a1c7329f31f91[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Dec 1 17:52:16 2020 -0800

    Resolve black error on activations
    
    
    Former-commit-id: a3ed18e33763313f109adba3970c5c1dcff074fa

[33mcommit 46c0057a2bf353abe4e2ef8ca977de60cdad683e[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Dec 1 17:42:37 2020 -0800

    Refactor and train Forcenet successfully
    
    
    Former-commit-id: 1af42860499d24cc7510c9ef27d8cb0ac4191bf1

[33mcommit 469f32da5742b5bd87816e847d19930ae9e2926a[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Fri Nov 20 17:31:53 2020 -0800

    Add initial version of ForceNet
    
    
    Former-commit-id: fb7459ec0875d179c8c9efde4419b19733566e7a

[33mcommit 75ab0293ab933e90b6df524499ceb0bb229973d2[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Nov 19 14:04:21 2020 -0600

    bugfix: ocp-ase calc support (#148)
    
    
    
    Former-commit-id: 6128d7ed3af32e096478aea2f29f3a3ec88f19ea

[33mcommit 8a1bd767122931a42ff84229c67abed302f3871d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Nov 11 18:44:10 2020 -0500

    Adds support for `otf_graph` to DimeNet++ (#147)
    
    
    
    Former-commit-id: 369687e2a93a6bdcd5f92698d0de618bd4a3c792

[33mcommit aee9095b973ac272302753f31d3964af62d8ed21[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Nov 10 21:45:18 2020 -0500

    Implements DimeNet++ (#143)
    
    * Implements DimeNet++
    
    * Add pretrained models
    
    This PR adds links for different pretrained models.
    
    * Add description in readme
    
    * Update models.md
    
    * Upper-case markdown filenames
    
    * Acknowledges DimeNet official repo in the readme
    
    Co-authored-by: Siddharth Goyal <sidgoyal@fb.com>
    Former-commit-id: b443aeba617b7261a50e2a38aca43629f990d7a8

[33mcommit 9daf142828eb60117b91a4ea6db21c8c35087333[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Nov 10 18:34:21 2020 -0800

    Upper-case markdown filenames
    
    
    Former-commit-id: 345d3a56ca25b43740b178893560624b4eaff542

[33mcommit af90e9774284a74472f1412dbb31bc7b98e03600[m
Merge: f6ed4cf 8312a05
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Nov 10 18:24:08 2020 -0800

    Merge pull request #144 from Open-Catalyst-Project/pretrained_models
    
    Add pretrained models
    
    Former-commit-id: 468b307e0ea82b25825680865b39147d876c5826

[33mcommit 8312a05f81d4d6f05ac92f2afb5fc4c830743207[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Nov 10 18:19:41 2020 -0800

    Update models.md
    
    Former-commit-id: 3af321db27d3622f77dee81c112f842de11947e0

[33mcommit d94779473134fc6fa61704285edafd8b1cd96bdf[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Nov 10 18:10:57 2020 -0800

    Add description in readme
    
    
    Former-commit-id: 90f196607d2c5e0500b894382830275cd235d12b

[33mcommit c6aa23d14105124e8d787408993c1e0597ce6a81[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Nov 10 18:06:42 2020 -0800

    Add pretrained models
    
    This PR adds links for different pretrained models.
    
    Former-commit-id: 6ec80e959d472def55d54e5fc08f3a5b9349cbb7

[33mcommit f6ed4cf4c443cce088b595952d814bc5d62919e2[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Nov 3 12:12:42 2020 -0500

    predict call on test set instead of validate (#142)
    
    
    
    Former-commit-id: 46714104dc3b4c42ddf7c65a1e3bf893c1536988

[33mcommit 99cb427c9de08e9b6d0599fd6301ac28af67b956[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Oct 30 19:58:05 2020 -0400

    Casts S2EF predictions to float16; uses savez_compressed instead of savez (#140)
    
    
    
    Former-commit-id: 92fcc86e883617bf01073c7558b39e57569cda20

[33mcommit 7f07a8b20aa79f2fc7cb73f4571d70a96584833d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Oct 30 19:56:38 2020 -0400

    Deduplicates predictions before saving; fixes #138 (#141)
    
    * Deduplicates relaxation predictions before saving; refs #138
    
    * Fixes typo
    
    * Fixes duplicated predictions issue for other trainers as well
    
    Former-commit-id: 1db694ba1aa0e9ce2ad869500f39a7e34413d4a7

[33mcommit e883e62a87d7cb63b7e7756e992ea6cfb65a527c[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 27 19:44:08 2020 -0400

    all ranks use master rank timestamps/directories (#137)
    
    * all ranks use master rank timestamps/directories
    
    * cuda->to
    
    * adds broadcast instead of all_gather
    
    Former-commit-id: 26bb242ccb1e277bbd2f0857d91cd20db05936f9

[33mcommit b8343bbd56850106e7b94a3e5bbe06f173e43e6c[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Oct 26 14:47:52 2020 -0400

    reverts Path to string (#136)
    
    * reverts Path to string
    
    * cast Path as string
    
    Former-commit-id: 30507a514afcea7ad04a8bc38966c91e040eaf9f

[33mcommit 731c53e0aa6cb50ac2ab62c1d49793ac79453612[m
Merge: 5626f72 b255f6a
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 22:53:34 2020 -0700

    Merge pull request #135 from Open-Catalyst-Project/trainer_refactor
    
    Refactored trainers
    
    Former-commit-id: 1f6d23b54850d818ca8d8f94533d64bd657d296f

[33mcommit b255f6a528a0c0a535111869d155f857117b3ca4[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 14:08:35 2020 -0700

    Refactored save_results to BaseTrainer
    
    
    Former-commit-id: 00d91f6761dcc0b038b3498781fac86c64abd9a5

[33mcommit d880064dbe2d0b6d479b6de26831227204586ec3[m
Merge: d172edb eda8845
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 01:22:04 2020 -0700

    Merge branch 'trainer_refactor' of github.com:Open-Catalyst-Project/ocp into trainer_refactor
    
    
    Former-commit-id: b641ca65e098d350b6cb004db00e0cc53222f555

[33mcommit d172edb6255cbaebd0662ad27402779ae2a00ddd[m
Merge: 1def74e fa2bda0
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 01:16:50 2020 -0700

    Merge branch 'trainer_refactor' of github.com:Open-Catalyst-Project/ocp into trainer_refactor
    
    
    Former-commit-id: 2875e408213787f51fd7742c17645e811ce1dd0b

[33mcommit eda884570da911323b8c122dd82ffc65dd7dc981[m
Merge: 1def74e fa2bda0
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 01:16:50 2020 -0700

    Merge branch 'trainer_refactor' of github.com:Open-Catalyst-Project/ocp into trainer_refactor
    
    
    Former-commit-id: 3b8d8874106bac60c6eb008be80ae765ec47b95f

[33mcommit 1def74e5a83fce73529ef4678e39b30561f3a762[m
Merge: de77ed1 b60a484
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 01:15:19 2020 -0700

    Merge branch 'trainer_refactor' of github.com:Open-Catalyst-Project/ocp into trainer_refactor
    
    
    Former-commit-id: 02b6a56a6d0e8160334a759efa9151b5d69ccc9d

[33mcommit fa2bda0e52d5b6111c5c6debd08962f275e185b0[m
Merge: de77ed1 b60a484
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 01:15:19 2020 -0700

    Merge branch 'trainer_refactor' of github.com:Open-Catalyst-Project/ocp into trainer_refactor
    
    
    Former-commit-id: ac73296b4b107fdb0cd627756fdb989141ccbdc4

[33mcommit de77ed13af1b363a525073879b6fb4e941b8f34a[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 00:16:16 2020 -0700

    Refactored trainers
    
    
    Former-commit-id: 1388a39e77af61b477548c0d39983aced894604d

[33mcommit b60a4845bd0266949bc1280ea8c7685589fdbf8f[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 25 00:16:16 2020 -0700

    Refactored trainers
    
    
    Former-commit-id: ba9c216ec017ce3dda140f2a3a4514af43af6fa0

[33mcommit 5626f72f15113ac0e38780bdc6cdff1741b8fa7e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 20 16:47:21 2020 -0400

    removes commas from config example (#133)
    
    
    
    Former-commit-id: 714da2b2e224296030acf42d3ecf7a005ef3b268

[33mcommit 4ded5cb9550d97a1d27474de75d5baaf4a1381ee[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 20 09:32:49 2020 -0400

    Update README.md
    
    Former-commit-id: 395d4e95855c3b74beb2442f37ec681860bfb360

[33mcommit d24fc7dd2800670cedbcd18a6a32f47e35c2e046[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Oct 19 19:51:20 2020 -0400

    fix - resolves ddp prediction writing. json -> npz format (#129)
    
    * fix - resolves ddp prediction saving. json -> npy format
    
    * store as npz
    
    * adds ids to prediction pos file
    
    * combines multi-gpu generated predictions at predict time
    
    Former-commit-id: 99ecaa29ed6655e771948b3f08624d9020bca5a2

[33mcommit c8987b652381ce71a7199fbfde97fc1ee9aeac7a[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Mon Oct 19 15:51:38 2020 -0700

    Add test sets for S2EF and IS2RE/IS2RS tasks (#132)
    
    
    
    Former-commit-id: 071b2cf8ccf6b6be769ad4af6b60744c979bb9b4

[33mcommit b7a40faee7765734091e084ac619a54ab0ab0604[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Oct 19 13:43:58 2020 -0400

    adds sampler to relaxation loader (#131)
    
    
    
    Former-commit-id: e7d7b4527426ac0a51b69e8ffffc65b2d9a1d2d1

[33mcommit e7b07aff5118bf4da3e7eff1673f57d71b1c7d80[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Oct 16 18:28:16 2020 -0400

    resolves uneven chunking (#130)
    
    
    
    Former-commit-id: 244e5491534f50ad78521ec688f6000bcb063c0e

[33mcommit c370def56e2043ee3d1b538611a24a4a050e6d51[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Oct 16 09:13:16 2020 -0400

    stores system ids as keys for jsons (#127)
    
    * adds progress bars to preproc scripts and test-data support
    
    * pin_memory control from config
    
    * prediction jsons saved by system-ids
    
    * remove extra bits
    
    * removes extra
    
    * save forces on free atoms only
    
    * ids -> id
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 4071947c598d296621161684a241302424d1eb45

[33mcommit 29074d700adc19d4588f4d256528b9afe197a243[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Oct 16 02:08:59 2020 -0700

    Fixes the iteration count used for logging to tensorboard (#128)
    
    * Fixes the step number used for logging to tensorboard
    
    * Trying to get the build to pass
    
    Former-commit-id: 0814da36d144aeb3b833778ff24b15d3d0a12f47

[33mcommit d129121d8ff8f31b0db40671ba395516f4f5256d[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Oct 14 22:37:30 2020 -0400

    updates training tutorial per dataset paths (#125)
    
    
    
    Former-commit-id: 49520d9936b97178a40cae3415847721aabd4c62

[33mcommit e702fc97438e02ddd6a50dcc57e78fd1cc37f8db[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Wed Oct 14 14:44:54 2020 -0700

    Demo notebooks (#123)
    
    * schnet example notebook added and config documentation started
    
    * removed config.md
    
    * updated schnet example notebook
    
    * updates schnet training notebook
    
    * update readthedocs
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: e294c9d5cac06f51d40ccd1744a0e0f9480c86e9

[33mcommit bf1762d67f10b81d6266d363db2e1ebf8bd6c777[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Oct 14 17:36:28 2020 -0400

    trainer typo, tqdm control (#122)
    
    * bugfix
    
    * adds tqdm control. distributed sampler for relax loader
    
    Former-commit-id: 98ebfecd5fc6f43c45fccfc52158afd7aadbad63

[33mcommit c6d3fc014523d442fb473f2358d4ad62dbc8b3e2[m
Author: clz55 <52088675+clz55@users.noreply.github.com>
Date:   Wed Oct 14 06:06:18 2020 -0700

    Update DATASET.md
    
    Added OC20 dataset terms of use.
    
    Former-commit-id: 7400f42e59c01b14b5b5055e1f2cc7b7e9eda3b4

[33mcommit ed036776dea802218940296b2dad83e59ad975e3[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Oct 14 08:16:52 2020 -0400

    fix random seed in tests (#124)
    
    
    
    Former-commit-id: fe86c335af460a52ee7761f30aa3601e1ebb94d3

[33mcommit 34d3fadc1b9cfe6c660ecf00867dacba8af14961[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Oct 14 01:24:46 2020 -0700

    Fixes typo in website url
    
    
    Former-commit-id: 65dfa13c99d2628719db1d3a22ec237dd6100c1a

[33mcommit 2844b75588cb7b016e00845d034cc3f5922c8537[m
Merge: 4532665 f9e0e63
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 22:16:57 2020 -0700

    Merge pull request #121 from Open-Catalyst-Project/update_trainmd
    
    Update train.md
    
    Former-commit-id: 5e615f5d783f77fecefe89a84460369969a3c324

[33mcommit f9e0e637545906a55845a7272bbd319c6f2b8661[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 22:15:44 2020 -0700

    Update train.md
    
    Former-commit-id: 444317f517274a3f771d9f4cebe238b660e9a88e

[33mcommit 45326655d41d7842ec7aaf0b80bc31a185a4d77e[m
Merge: d8fb140 2c4c538
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 22:07:47 2020 -0700

    Merge pull request #120 from Open-Catalyst-Project/update_targz
    
    Update dataset.md
    
    Former-commit-id: b159feb5e3d641b7b8f4b067a59a2c71d4a4a65f

[33mcommit 2c4c5384cc2c995bde032a99862e465dcf1d5969[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 22:05:13 2020 -0700

    Update dataset.md
    
    Former-commit-id: 0ad9370e748fcd73361299d4fc1eff97b8b7e3fb

[33mcommit d8fb140d88adb61d797206866c74e775dfaf5c67[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Oct 14 00:59:09 2020 -0400

    remove unused import (#119)
    
    
    
    Former-commit-id: b4538fc0cb9024d81788a89405a482176fc50503

[33mcommit 8ad4bb776041c6cb9705e9f1ba78fa9b967d59be[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:55:47 2020 -0700

    Updated docs to match readme.md
    
    
    Former-commit-id: 26f396fb58d8d1507eb05cdb2d59b3c99ae8a42f

[33mcommit 90c15b92a9def4da460c2c6140bee74d8cf892b0[m
Merge: c24c783 ba4f498
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:49:54 2020 -0700

    Merge branch 'master' of github.com:Open-Catalyst-Project/ocp into master
    
    
    Former-commit-id: ca121cdb3a8148bc62b47a439b38c0402e22bc22

[33mcommit c24c783052a93afb42a793df7d956fe0637dc007[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:49:20 2020 -0700

    Updated docs
    
    
    Former-commit-id: 0951a1f053a7fcf8389d34fb282717e4b8b14d7b

[33mcommit 0597161c469d614e9cc22dc07285434dbad6a97d[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:49:07 2020 -0700

    Updated docs
    
    
    Former-commit-id: 843c0007dc3fca195d7789e018666b350fcd4c9a

[33mcommit ba4f49838a39e8f255086810568e018d6ad0dd84[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Oct 14 00:43:36 2020 -0400

    repo clean up (#118)
    
    * repo clean up
    
    * remove is* preproc script
    
    * black
    
    Former-commit-id: 4f5c1b7c5a9b0ae3afccd69c2f5e614ebc54ea7b

[33mcommit 463f09a23f61b134430e32088bf132e68e822cd3[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Oct 14 04:42:53 2020 +0000

    small clarifications in data notebook
    
    
    Former-commit-id: 480752fcfed4a64d01f3378eda72815dbdac43be

[33mcommit 6d57a3bbd9b40b503c7784ca64656a687ba008e6[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Oct 14 04:22:02 2020 +0000

    baselines->ocp
    
    
    Former-commit-id: 5b4cacfd6df31712856905cb890abbdd3ee5c3bc

[33mcommit 29b5d6b5ea73a53eb37c824efa5697785031a56f[m
Merge: abee72b 6c6950b
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Oct 14 04:18:31 2020 +0000

    Merge branch 'master' of github.com:Open-Catalyst-Project/ocp
    
    
    Former-commit-id: 827a25e444b186c972cd6528981b08af8ca02fb0

[33mcommit abee72ba1399e5a85a6e350e67da76cebbe1d62d[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Oct 14 04:18:26 2020 +0000

    apply black to conf
    
    
    Former-commit-id: ff12055cab58acad74c479ad453321dc9522acfa

[33mcommit 6c6950b1805b6eba39d42f1ce6aca6aa8499c6dc[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:14:25 2020 -0700

    RTF yaml file
    
    
    Former-commit-id: 2be158602fc0f68ce0149d54a10abb1549cd7bb8

[33mcommit 1caaa00784dd6a52fc120d7f08b4bcbd9c9b07f7[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:12:25 2020 -0700

    RTF yaml file
    
    
    Former-commit-id: 90308cc27c62b6f41fa05ca1ccb5e2fb83374b18

[33mcommit be45fe4f3cbc7237d2e3696bf75d7f9b83366ce5[m
Merge: 707eeae 9604e03
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:10:54 2020 -0700

    Merge branch 'master' of github.com:Open-Catalyst-Project/ocp into master
    
    
    Former-commit-id: c1b45eda715ffdbbb75b294b4f9cdacf71af30dd

[33mcommit 9604e03aee93a14e58e92ff5a061d62a6bd88a8c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 13 21:09:09 2020 -0700

    Fixes typos in configs (#117)
    
    
    
    Former-commit-id: 272540f72cbde51f266329a15fccb6ef14c8b3c9

[33mcommit 707eeae7362e0ea2003fff1cea4eb1b41b2ac2eb[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:08:52 2020 -0700

    RTF yaml file
    
    
    Former-commit-id: 71a8ef51c55ddb7717018d8b434d97fe1ca209e4

[33mcommit d1c12552072198d84c190c527ae524be50f345d8[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 21:04:55 2020 -0700

    Created env file for docs
    
    
    Former-commit-id: 9b0fb8584959b09732515a498d893230c07a0bcc

[33mcommit 3c6fa25118dd01c1f66daf685f8b3c94032fac64[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:57:06 2020 -0700

    Updated docs requirements file
    
    
    Former-commit-id: 9a99cc03660a036a1acab149c7df78d7d1097115

[33mcommit f7f1438a9a77fdcc8839999b3adbd1c7f52843ab[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:56:52 2020 -0700

    Updated docs requirements file
    
    
    Former-commit-id: 3c44fed5d2c5d9e2f7ffb808853562dcdd4e3972

[33mcommit 070bebe28bab4f737fff23e9dbbf4ab861b10c30[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:48:14 2020 -0700

    Fixed docs
    
    
    Former-commit-id: e027da6abf9d6dc9e8b6a77e241e5668aa24a1d9

[33mcommit f877b6bee43695afced328bda230d864dd18fdad[m
Merge: 3919346 ca6550c
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 20:41:13 2020 -0700

    Merge pull request #116 from Open-Catalyst-Project/fix_readme_with_dataset
    
    Update README with dataset page and project page
    
    Former-commit-id: 4b25cebef9ff35021125990907d1860eed0d677d

[33mcommit 3919346f5b3a7fe4ec1f6ea9b8a80dace76ef716[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:38:42 2020 -0700

    Update conf.py
    
    Former-commit-id: f06b483a29feb9170b6436be4a7710e311c9e462

[33mcommit ca6550cca1dc8924351fa61eee18c300f0fadf98[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 20:38:09 2020 -0700

    Update README with dataset page and project page
    
    
    Former-commit-id: 8c62415f10af467395e5283c5367d90a2587f233

[33mcommit 730fa910235a144b9af62af04708561487735407[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:34:26 2020 -0700

    Added a requirements file for readthedocs
    
    Former-commit-id: 266aca6bde67c6fd42b87f3994014664c41f9887

[33mcommit 48ea75264bf1b690fdfc8bc71e9252e65c2204e9[m
Merge: 3d8a1c2 c23b249
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:19:10 2020 -0700

    Merge branch 'master' into release
    
    
    Former-commit-id: 6f31539f3acb4532b7f9ffeae2f53c5597499952

[33mcommit 3d8a1c2b625c8ebe8112e5bd122364108add7acc[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 13 23:18:27 2020 -0400

    missed inits from efficient_validation branch (#115)
    
    
    
    Former-commit-id: 8ae974edc207092a7cfb1e5d7d1b088e5e2c5341

[33mcommit 240792dd6026ab2a44281c05d49c91094bbe7b6a[m
Merge: fe579b7 79ff3af
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:12:10 2020 -0700

    Merge pull request #109 from Open-Catalyst-Project/dataset_md
    
    Add dataset download page with links
    
    Former-commit-id: 93281f7bc8a6f238bba8e72ec6144cfede44bdd0

[33mcommit fe579b7a8ea32a76d1f77decdda7738700773d86[m
Merge: 22bca54 c7d9d80
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:11:52 2020 -0700

    Merge pull request #100 from Open-Catalyst-Project/release_documentation
    
    Sphinx documentation
    
    Former-commit-id: 1b567a2aa3f432288c63b160330f3e546657ba65

[33mcommit c7d9d80969b21a1d1a9e27f11316844d1a4ea419[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:10:07 2020 -0700

    Minor changes
    
    
    Former-commit-id: 25580d5e6df608803a42e19a8b62a8d21f7e01ed

[33mcommit 22bca5439c56cf2f70dd8aa01ef5e22423023c30[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Tue Oct 13 20:04:37 2020 -0700

    minor changes to the readme preprocess datasets section and space between $ echo (#114)
    
    
    
    Former-commit-id: e8cdb496c364d8d66aac3cdb44358f74f2e483f7

[33mcommit d133b1491ad39ff709153d92d0636675169a1665[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 20:01:23 2020 -0700

    Minor changes
    
    
    Former-commit-id: 6cb4da7bd3e783b55060589547b74a7477c88322

[33mcommit 03bb840adce5a948c0517c62d9c9ac88be7ed59a[m
Merge: 40d0ca1 bf08d1f
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 19:58:11 2020 -0700

    Merge remote-tracking branch 'origin/release' into release_documentation
    
    # Conflicts:
    #       train.md
    
    
    Former-commit-id: cb9b4b9232791bd0137e8e85037703ed8ba94620

[33mcommit bf08d1f5f40f52f4f3b79f5232f97b307f0cd589[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 13 22:56:17 2020 -0400

    adds updated ef preprocess script, flag for edge information in graph construction (#91)
    
    * adds updated ef preprocess script, includes flag for edge information in graph construction
    
    * variable name change
    
    * updates docstring to clarify for EF task only
    
    * adds uncompress script for ef task, adds assertion for otf computation
    
    * typo in preprocessing, remove tuple
    
    * otf support for LMDBs w/o edge information (#108)
    
    Former-commit-id: cf831a1bb435cc19e2d7ca3ffaaa8f77eb4c8374

[33mcommit 131b8542d1d5029f919465a6a9066340d097e79c[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 13 22:55:40 2020 -0400

    run-relaxation to detect val/test, small updates (#113)
    
    
    
    Former-commit-id: 8a907e1c16807d70fa92032cb285af41cafa9b0c

[33mcommit 40d0ca1a82dc2d6bf335a9b6028ce7e70f3d8840[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 19:49:28 2020 -0700

    Minor change to training.rst
    
    
    Former-commit-id: bbafc29cfccb12d30cdc94c9381fc1d2b07de8ac

[33mcommit af2f770a97a16fab9ff4aa1053759600a717929b[m
Merge: e3d740f 14adc31
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 19:44:11 2020 -0700

    Merged with release
    
    
    Former-commit-id: 3a4af9df936fa8ded497346a72df676485051fdb

[33mcommit 79ff3af070b18c474fec3122608a7d2082da9cb1[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 19:10:39 2020 -0700

    Add more links
    
    
    Former-commit-id: f23052562b0f9faf3b3996f599c2bb3133517cf5

[33mcommit 14adc315faf15f29d23d043f2ef673b46a236a9e[m
Merge: 6d9ff5e 5189be6
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 17:31:08 2020 -0700

    Merge pull request #111 from Open-Catalyst-Project/model-configs
    
    Adds all model configs
    
    Former-commit-id: 5fbecb6ad8b2814aebcd62662ed2323c4f4ee161

[33mcommit c23b2496546711f35628721fdc4d4bffd85d26de[m
Author: Devi Parikh <parikh@gatech.edu>
Date:   Tue Oct 13 17:29:19 2020 -0700

    Update README.md
    
    Renames Pythia to MMF
    
    Former-commit-id: e96a80b82760fe0fe35409180e97ec2a8d72ad2b

[33mcommit 6d9ff5e8352b911091ed87df279f3b00863625c2[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 17:29:13 2020 -0700

    Removed dimenet++
    
    
    Former-commit-id: c7d8e255f83ad7d2b9d524ea929b20c3e4933776

[33mcommit bcea7be22d3ed5413404f66861b3086f531a715f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 13 17:08:59 2020 -0700

    Save ids as well in init-to-final preproc (#112)
    
    
    
    Former-commit-id: 985fcc399d52fd10656c76ec570d067482405e80

[33mcommit 5189be6b439eafe600317c24140b65db1a8059c2[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 13 16:34:37 2020 -0700

    All model configs
    
    
    Former-commit-id: 8a53fda69facd8e557b691296a4bbc63d9245e5f

[33mcommit 1e782eb9dc13331cbb52fed9022ae0edb4ddd3b1[m
Author: Siddharth Goyal <sidgoyal@fb.com>
Date:   Tue Oct 13 16:17:44 2020 -0700

    Add dataset download page with links
    
    
    Former-commit-id: b3b1e0e37b96d01286ef39b301e34e005ec375ae

[33mcommit 938f35e09ffb5b67e068a4d4b1071960d91789fa[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 13 16:03:04 2020 -0400

    BUGFIX: broken trainer when val set not specified (#107)
    
    
    
    Former-commit-id: 590288ad63fbf702c4e0b13b4b8b1274c84b702f

[33mcommit e3d740f69a0594a45e68d01386ec82dba28a2baa[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 11:34:15 2020 -0700

    Updated env file
    
    
    Former-commit-id: dd5ab8a783fb4e0b1bc1d20f1693114f0e9038af

[33mcommit 7aebab6105bb1a40c2a571d5f97edafeb0b122dd[m
Merge: f698bed 90cb82f
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 11:22:35 2020 -0700

    Merge pull request #106 from Open-Catalyst-Project/e_trainer_normalizer_bug
    
    resolves bug in energy trainer predict
    
    Former-commit-id: 4cc4f67aeffbf50fd3e3d8c18d5be83e955dd0c7

[33mcommit 90cb82ffa0547f1c8e1d7ece08e377549ce4bf5f[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Oct 13 16:22:15 2020 +0000

    resolves bug in energy trainer predict
    
    
    Former-commit-id: bcc7ac2fcf59f9c161f0ad788d7fd1037fd60e29

[33mcommit f698bedd443c0f506836140d4c2ab22647eb7c0e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 13 12:17:38 2020 -0400

    Update train.md
    
    Former-commit-id: 221ed311d03496777e7de0b59b6ba9946187344d

[33mcommit ae1f0e08f664e43cafdf385a6653879a79762598[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Oct 13 10:53:08 2020 -0400

    defaults normalization to False. True requires additional params (#103)
    
    
    
    Former-commit-id: eb4e917d69325382e466ca9edf0af990caddba1c

[33mcommit ac8f1bea87a2f910b81f8e53a568c9cd1bf439f7[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 01:58:27 2020 -0700

    Added more documentation
    
    
    Former-commit-id: 88162e49513f15010ffd88140b09c86e4e7939ce

[33mcommit 675e47c0a0a5a6d4c15343e0c5adf6e5c197e28b[m
Merge: d527223 5042701
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 00:13:05 2020 -0700

    Merge branch 'release' into release_documentation
    
    
    Former-commit-id: ec66779628dae1e7d3305c544ecc092a9d4000e3

[33mcommit 5042701135ecbf8e6f8c5e70047b7bed5ac6f930[m
Merge: 47165b8 2b7b7b7
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 00:10:24 2020 -0700

    Merge pull request #99 from Open-Catalyst-Project/release_ci
    
    repo updates for successful CI build
    
    Former-commit-id: be40ed58ad99cabf02f2a01c87d325091cc9d6d2

[33mcommit 47165b8e815af593bdd8099789967a75ed7150a3[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Oct 13 00:00:11 2020 -0700

    Minor fixes
    
    
    Former-commit-id: a0eea36990e0cbbd712a9860b291373de60140ff

[33mcommit b68585b233216fa79b5cece71f0460983e1a7c08[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 12 22:40:28 2020 -0700

    Bugfix: load normalizer state dict on correct device (#101)
    
    
    
    Former-commit-id: 58baac63f24f719eb1b511b72a89b15c749fee48

[33mcommit d3764e40b95495eeeeedcb0119d120746123abfe[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 12 22:39:33 2020 -0700

    Bugfix: checkpoint IS2RE model from master process only (#102)
    
    
    
    Former-commit-id: 02224a479b2996743b4c52d542ea9bea99da7e16

[33mcommit d5272230b91de7a238565c2ae322b6355f897c88[m
Merge: 010a4eb e38a337
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 18:11:45 2020 -0700

    merge
    
    
    Former-commit-id: 8b8936e33746abad4281039f49afb172b692beac

[33mcommit 010a4eb64e1a88be5f8e9aa2c006c0cc701af76d[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 18:06:28 2020 -0700

    Added sphinx documentation
    
    
    Former-commit-id: 4caf14f161df99109c40baf439fb94bc5d4389cd

[33mcommit e38a337bb27a247487baf66a996f4c6e0f41fbbd[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Oct 12 13:13:15 2020 -0400

    corrects dataset usage
    
    Former-commit-id: afa05e47c6efccaa75fbce844fc55fa1d4eaa384

[33mcommit a9c9180c5fd4855748c67c6cf345c8d154992783[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Oct 12 13:06:50 2020 -0400

    Update train.md
    
    Former-commit-id: b641bee77b519d4ea523bfb18d65893fb021ee73

[33mcommit 4fee368b387596ee7763b291d09445c274ae3645[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Oct 12 12:55:48 2020 -0400

    Update README.md
    
    Former-commit-id: df7eaac896e038aed1c5dec0d8d43b3f85b6e07e

[33mcommit 2b7b7b707a57c3f025377893e3cdb6b27b417db2[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 15:40:17 2020 +0000

    update cache version
    
    
    Former-commit-id: 2eaaaa71897b8bbaf0c7594afac10369704610b7

[33mcommit dfc929fcd9a823a998c0289f7b23366f1a383c73[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 15:06:52 2020 +0000

    remove isort for ci build
    
    
    Former-commit-id: 80aaa72713ea1eb06ca212eadd3cf0899682b592

[33mcommit a11f263cd8199e8e6ec90a1a75a3d25db59a8d21[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 14:46:43 2020 +0000

    circleci config typo
    
    
    Former-commit-id: 370332afc17ae70fff75ddb1ebea9a44abcd868b

[33mcommit 25956fb5feb73552aaa3c911633ad03102f3cc71[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 14:27:32 2020 +0000

    apply black/isort to repo. Adds isort to circleci build
    
    
    Former-commit-id: 316ec9e43d62b9fe9032a551e6f87a80bca6f069

[33mcommit 2e44cdb8f75134126b7b4a0d9e3644af0ed98701[m
Merge: a75644f c018ddd
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 13:38:29 2020 +0000

    Merge branch 'release' into release_ci
    
    
    Former-commit-id: 92ca87b3f7977f04abd7fbbf2616ec0717954e12

[33mcommit a75644fbd044cf96bbea6a5fa84ae66848d6c9db[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 13:30:52 2020 +0000

    debug circleci
    
    
    Former-commit-id: fbb18e875bd07034c67225126ae123fd2106f941

[33mcommit c018ddd13c6edf2e1c096112d7cfc7b1cd903fab[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 13:23:12 2020 +0000

    updates dependencies for cpu only env
    
    
    Former-commit-id: 1d3c441307aa7b55030a40cf67b183743fd231ec

[33mcommit 9c0abcacbd6b79ed5e28fff865fba51d1df233b0[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 12:43:34 2020 +0000

    empty commit for circle ci detection
    
    
    Former-commit-id: 73a79daced76528dcfcb48ff666c5779fa10a920

[33mcommit 9843dc9c499f7e4b6f6786aa5194fa312e179eb3[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 05:35:18 2020 -0700

    Added Circle CI config (#98)
    
    
    
    Former-commit-id: d7446f48a44471144cc6ec992f8c7f8908c64840

[33mcommit d0e7349b065f607bd066bf479e282acc45547a72[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 04:29:28 2020 -0700

    Added Circle CI config
    
    
    Former-commit-id: 627fd302abf78bbe4ce835e2beb965b2ea008dab

[33mcommit 158573a55f5f1738565f98f454c9a8a436b4a616[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 04:22:35 2020 -0700

    Sphinx documentation
    
    
    Former-commit-id: b9db0254d299a1e9e56d3df7dd7cbcecb45d317d

[33mcommit 679d9f027b4f7e4938a1051f8a5421df50737e01[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 03:28:59 2020 -0700

    Added docstrings for all the data loaders, models and trainers
    
    
    Former-commit-id: 33e290903b8178c6d2b4b8cb9530bf228eb8c41e

[33mcommit 8390c0decc6849982d37ffeb9ec286008fc25a60[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 02:25:48 2020 -0700

    Revert changes to main.py
    
    
    Former-commit-id: 613fa44d26e0eb2a8c9ffbf0ec1dfe727a3eb968

[33mcommit e977ee9949d01181977f39d12fa24e57a25d2b2e[m
Merge: d15fc8e 65f6298
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 02:23:31 2020 -0700

    Merge branch 'release' of github.com:Open-Catalyst-Project/baselines into release
    
    
    Former-commit-id: e5dcaffc4edf46de363daa74033685c015110e8d

[33mcommit d15fc8ee8fa46248561b4c59653a5deb09ff9e01[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 02:23:20 2020 -0700

    Changed type of flag to str
    
    
    Former-commit-id: c5fd5ba537de2c916af1a4cef25c8376d6bf86aa

[33mcommit 65f62983d9a86ed4b606f6398375a31a539715c9[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 02:19:41 2020 -0700

    Changed type of flag to str
    
    
    Former-commit-id: 62887ce1087efa141316e3440666c4f79b735852

[33mcommit 920f11e798dba8f71152fe1c7fa39f3f92511517[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 01:07:52 2020 -0700

    Added license information
    
    
    Former-commit-id: a708db8c9fc9986e70750e95fbb05102f77d9a76

[33mcommit 9a0f2d1434c0b00e77ff4a2e15cc586ee8bbf943[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 12 01:03:21 2020 -0700

    Config changes
    
    
    Former-commit-id: 9b6ca05ff92e9233c6213662b6ba499dec850ee4

[33mcommit ec17c1bbc22fa5b422f6cc843dbcca4f8121ca9d[m
Merge: c11a717 8656482
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 21:43:27 2020 -0700

    Merge pull request #92 from Open-Catalyst-Project/release_train
    
    [Release] Merge efficient_validation, add training tutorial, support predictions and relaxations in main.py
    
    Former-commit-id: 3a03dafaf1222b2faf002f36d55c957859c5f008

[33mcommit 8656482c5563a172b5128b4d8cba9cb867935e52[m
Merge: d752393 25e67e8
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 18:47:49 2020 -0700

    Merge pull request #97 from Open-Catalyst-Project/rename
    
    more consistent naming
    
    Former-commit-id: e756e56ee03b24347fddf203a4fca19b7f768429

[33mcommit d752393c16f17ca2a246672b2740da4c726eb611[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Oct 12 01:43:16 2020 +0000

    small fix to predict function
    
    
    Former-commit-id: d50db95e41f143985aaf222ae943c1d09dd88906

[33mcommit 25e67e87bdbe058a3516c44869a4545ee4217ac5[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Oct 11 23:09:40 2020 +0000

    more consistent naming
    
    
    Former-commit-id: caa319ae33318148ccf5d4f1997c5b228ba390cc

[33mcommit b1a81d5ff0c1e8f2ae87d9dbb83e70b5d802d555[m
Merge: 7be2f9e f7245ef
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 15:54:45 2020 -0700

    Merge pull request #96 from Open-Catalyst-Project/readme
    
    updates readme
    
    Former-commit-id: aee42378a617b33d298ad7c05af5db17b703c843

[33mcommit f7245ef70b2f388cab715fcbc8c7a82c3938ea57[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Sun Oct 11 18:52:33 2020 -0400

    formatting typo
    
    Former-commit-id: 71793bff39e0ae2f9ee3a7d9cb46ad3b6a888dcd

[33mcommit 303717a857acd85725cf5296ad1e7903a2afc422[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Oct 11 22:43:31 2020 +0000

    updates readme with dataset
    
    
    Former-commit-id: cfd253e145d10bed860ee857af7ceafa5c04458b

[33mcommit c11a71702dbcdb06763a8ba0000da5efbff89b73[m
Merge: 2672ecf c8c1033
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 14:41:35 2020 -0700

    Merge pull request #90 from Open-Catalyst-Project/data_viz
    
    initial data playground notebook
    
    Former-commit-id: 49b3cc460be5f1f947b873c60bd43bedcaf63d36

[33mcommit 7be2f9e6aa327377ec5b68a7094c24be71babd13[m
Merge: ac6e1ae 8e662c1
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 13:55:20 2020 -0700

    Merge pull request #93 from Open-Catalyst-Project/eval_format
    
    save prediction files in format for evalAI
    
    Former-commit-id: a46db70dab1416988a9ea83818e5c48b98ca72f0

[33mcommit ac6e1ae407e768aa964b9aacdad8812e5acac04b[m
Merge: 500cb2d bdcef5c
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 13:39:17 2020 -0700

    Merge pull request #94 from Open-Catalyst-Project/test_eval_fix
    
    updates is2rs eval test
    
    Former-commit-id: 79b056d6043aabb580ebdbbc54ce61e4e98f9ab9

[33mcommit bdcef5cf31ba17beda4e57bec0d463aa10f30071[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Oct 11 20:37:41 2020 +0000

    updates is2rs eval test
    
    
    Former-commit-id: 504525e66a78c1bc6f1245f9a1aab5ac0cd8fc6d

[33mcommit 8e662c1c720d189e1f0d96987b1f696819af649a[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Sun Oct 11 16:14:44 2020 -0400

    updates train.md for evalAI format
    
    Former-commit-id: 009b6197e6f11f2349a105ce094e69e7c9787882

[33mcommit c338dad0eecf31755c8a4f6a46eb870f694727dc[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Oct 11 20:14:10 2020 +0000

    makes split names consistent with paper
    
    
    Former-commit-id: ca2bd8d48ccda7f1de3fb30c0373f84be8f767f8

[33mcommit aec039914342d0c6c5afc96527ef672e8796cf24[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Oct 11 19:55:26 2020 +0000

    adds script to merge relaxed_pos.json files into a single file
    
    
    Former-commit-id: e44717293d6047c07f57ef533ac95a11ee441c21

[33mcommit cd7a25d14487ae672abb1d2b49f3e6f0812cb7be[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Oct 11 17:58:19 2020 +0000

    typo
    
    
    Former-commit-id: 2a1cc60bad133f076592faaeced9cd0fedd86981

[33mcommit 237c144af161e9ccaabcc21844faf28d22a363d0[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sun Oct 11 16:50:44 2020 +0000

    save prediction files in format for evalAI
    
    
    Former-commit-id: 31fa2b4f3ffc123c92f242bb8d9a3255d02b846e

[33mcommit 500cb2db53da32e6d1c0751ab388c952f9a6ef01[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 03:48:26 2020 -0700

    Added OTF computation to CGCNN and Dimenet
    
    
    Former-commit-id: f4702f81b9bd61e0702e268de1d4392393168faa

[33mcommit 3c6e8dde390222e81feca47f60103b23f6c16444[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 03:44:35 2020 -0700

    Removed lmdb config file
    
    
    Former-commit-id: 45dcafc6145fea621ec8dcfd265e73719b3d3858

[33mcommit f46fb7fca8db4177f7b56e74382c891455e27c60[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 03:31:33 2020 -0700

    Added training documentation file
    
    
    Former-commit-id: b8d4853a55a5b1ab10bf2bc0ec86aba20f70f08a

[33mcommit ee18c8c81ca27d766b0f398fe76c06669ee65aab[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 02:32:41 2020 -0700

    Support for running relaxations from main.py
    
    
    Former-commit-id: fd5c3849c2c0cdd2fa68dcc75839676f02408d26

[33mcommit 8d3f4e25280b79eeb5ea993f1a26d9e92e531853[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 01:37:26 2020 -0700

    Merging with efficient_validation branch
    
    
    Former-commit-id: b042ed37ed7c0b09d914c7d516c6ee84b4573d44

[33mcommit 4b019cc815fd454ff95a68f35f37bf1b342ce2b6[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 01:18:11 2020 -0700

    Support for relaxations from main.py
    
    
    Former-commit-id: 178f323163e2acc60c78569ccca813149dc9ce51

[33mcommit 06d644427ab7bb590a8bc66db6dd317967a39a21[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 11 01:04:44 2020 -0700

    Support for predicting using main.py
    
    
    Former-commit-id: 1d4a6e0f03dbe0f778e7557b9ce643b8b152eb92

[33mcommit 16513d483d8a250fb51872d36b63b461bbaa5af7[m
Merge: 50b0b8e 2672ecf
Author: anuroopsriram <anuroops@fb.com>
Date:   Sat Oct 10 23:14:23 2020 -0700

    Merge branch 'release' into release_runs
    
    
    Former-commit-id: a4ac25776ac571d62abb6580e07e169cc08640b8

[33mcommit 50b0b8e389e2b1eb7729aab06c0408fbc4848857[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sat Oct 10 23:10:58 2020 -0700

    Runs
    
    
    Former-commit-id: 487f2e2ac191acae639c83f518b106ce08f2d7e0

[33mcommit c8c1033643859ccd7de023ceab93c207f2b2a3bc[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sat Oct 10 18:34:31 2020 +0000

    plot sizing fix
    
    
    Former-commit-id: a718b6dd2c503a2f6b8da12baef97de173266846

[33mcommit d2c3af4cc63f2b8f31dbdc7cf270bb63a0dc461f[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sat Oct 10 18:31:20 2020 +0000

    adds energy profile visual
    
    
    Former-commit-id: 7224834d941ebcd142244d3cd50e42cba1b4ef5f

[33mcommit 383ba80f05259076d7b398b09d69150581339747[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sat Oct 10 18:24:28 2020 +0000

    initial data play ground notebook
    
    
    Former-commit-id: 99a59e34b5d1c5ddf2ca10eb4dd75cb2f5a2ad45

[33mcommit 2672ecf17eaaefb9dc177804578dccc8a0e7d6a7[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Fri Oct 9 15:00:00 2020 +0000

    remove more unused notebooks, scripts, and directories
    
    
    Former-commit-id: 2280e3f4b0902be4cd3f7209b3f0357aa35e9446

[33mcommit 6de1b9c8875afdc7c304b7ba2dfcae215c10a66a[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Oct 9 05:10:00 2020 -0700

    Removed old models, datasets, trainers and configs
    
    
    Former-commit-id: cb188f3dbc54aacbfe9bde53f4fe8fc444972284

[33mcommit 9391c7d45086df40926731c8f60d725648b4523f[m
Merge: 5ef9680 b37c2ad
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Oct 9 04:10:56 2020 -0700

    Merge pull request #88 from Open-Catalyst-Project/ddp_refactor2
    
    Allow non-distributed training
    
    Former-commit-id: 46b5d73685ea2384ee515662d463866729ea2070

[33mcommit 5ef9680e3ff1b9171e03bd4d3d3d11e33469858d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 8 17:25:17 2020 -0700

    Adds option to specify primary metric in config (useful for force-only models) (#89)
    
    
    
    Former-commit-id: bff757afa45386705812621232d94e8d41f81da0

[33mcommit 66fc4ea03308e979e8382d4ca183ab3f8c9f6831[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 8 16:51:04 2020 -0700

    Make sure force trainer checkpoints at least once
    
    
    Former-commit-id: 3943f8df6475b3541ab6ee4a141762e861b5e350

[33mcommit b37c2ad6c7fa402299d590d75212865fb3151668[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Oct 5 02:00:54 2020 -0700

    Add back --distributed flag
    
    
    Former-commit-id: 1185627b5b224c28169699bd72fce850a8d2daa0

[33mcommit a5ebe44a9e893a8ed716679fca5e9aa952b1902f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 4 23:29:03 2020 -0700

    Updates primary metric for s2ef
    
    
    Former-commit-id: 24b4ab172f427bbe9412cbeff50ca1a477ba4d2f

[33mcommit 2877804970282d6339d4a25173fda1967a4ae104[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 4 20:23:23 2020 -0700

    Add back --distributed flag
    
    
    Former-commit-id: 04afca6f2e6754620c034d565ff41cb4de518c76

[33mcommit 792a439fd69eed7e7e404b68c9d0d2e2afe25373[m
Merge: ed81d18 61399b4
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 4 20:14:43 2020 -0700

    Merge branch 'master' into ddp_refactor2
    
    
    Former-commit-id: f8a37337dc078c8421884e72d7af697ac64b4f56

[33mcommit ed81d182c0aa29c5cffe5a2b115623e448174421[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Oct 4 20:13:47 2020 -0700

    Allow non-distributed training
    
    
    Former-commit-id: 606834cf55208a3e9546c60c0f2b811ee5cc7137

[33mcommit 61399b4f55cc58f275b5b2907ffbbc83c051491a[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Sun Oct 4 10:56:33 2020 -0400

    remove distributed flag (#86)
    
    
    
    Former-commit-id: f932309747c3652f4850da9bb77437ddd25dda5d

[33mcommit 1756edd8729c186c7690218cef68eb8c138b2faf[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Oct 2 13:33:12 2020 -0400

    resolves natoms evaluator bug (#85)
    
    
    
    Former-commit-id: 2051faef1497f48f14d10dfb49f0bcd4f7c03b26

[33mcommit fb7cbe2ee016f3979c3cb69f44783941118c4c07[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Oct 2 01:15:02 2020 -0700

    Implements DimeNet++ (#77)
    
    * Implements DimeNet++
    
    * Updated config
    
    Former-commit-id: f311887658ec6ee61c79466223a2542656a85edd

[33mcommit 1ebc1006427d8c811023646290cd8e84869c71d4[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Oct 2 01:13:35 2020 -0700

    Removed non-DDP Energy and Forces trainers (#84)
    
    * Removed non-ddp Energy and Forces Trainers
    
    * Removed dist_force trainer
    
    Former-commit-id: d2357aa46e9bb519b5c0cdeb2c2c43a35655d087

[33mcommit ac9d84539d430e9e635b744f9ac63e410def2b8e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Sep 25 00:22:45 2020 -0700

    Minor updates in trainers and preprocessing (#82)
    
    * Minor fixes
    
    * Updated paths for final data preprocessing
    
    Former-commit-id: 345ecfa280fec1ca7e72211fcba80d172457bbed

[33mcommit 956e5d6753c6d605b6f6295a33bb355376b709e4[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 24 19:23:44 2020 -0700

    EFwT for S2EF and EwT for IS2RE (#83)
    
    * EFwT for S2EF
    
    * EwT for IS2RE
    
    Former-commit-id: b342c527851d43e5d7e97146ba6bb3e26f595a71

[33mcommit 2550755b83724269cd893bc6156d13be5139c678[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Sun Sep 20 18:17:46 2020 -0700

    Fixed data loader issue
    
    
    Former-commit-id: c44c749c284c75522a1808c6e5f2446e4d758fc3

[33mcommit e8b81b862731b1bd8770d67042eacc84485bde54[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Sep 9 17:48:38 2020 -0500

    bug fix with loading ddp trained model in forces trainer (#76)
    
    * bug fix with loading ddp trained model in forces trainer
    
    * clarifies argument naming
    
    Former-commit-id: 6cdaee26bd613ab91d97c7b2972a53b8866c1a97

[33mcommit b027a12ab7a21373ebb1a8799ed3e8eb25c4c162[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Sep 9 15:30:42 2020 -0700

    Fixes bug in computing angles in DimeNet (#78)
    
    
    
    Former-commit-id: e4433b05b2de48755168c2414dcd64293eab371b

[33mcommit 0032df9b7a09a1c3dc6b758270b9fb57a2774545[m
Merge: 66609d5 3bf0bb2
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:58:19 2020 -0700

    Merge pull request #69 from Open-Catalyst-Project/amp2
    
    Support for mixed-precision training + DDP for Energy Trainer + other optimizations
    
    Former-commit-id: a9b2e50fa7388825d880fdfdda10532f1fd43929

[33mcommit 3bf0bb22bfbe84ba8b3f9d237b13ba1fba523e6c[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:56:02 2020 -0700

    merge
    
    
    Former-commit-id: bed46dce046cdc3f7d62b773c4af667062dbf60a

[33mcommit ba74bfafde20d0cdded4e7aff3e761472dea7e7e[m
Merge: 72acf49 f82e163
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:48:33 2020 -0700

    Merge remote-tracking branch 'origin/amp2' into amp2
    
    # Conflicts:
    #       ocpmodels/trainers/dist_energy_trainer.py
    
    
    Former-commit-id: f051caedc9c63caf7a61683acea3ea7270878079

[33mcommit 72acf49e64b597edea301603de3a5f40d425d137[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:45:26 2020 -0700

    Added evaluator to DistEnergyTrainer
    
    
    Former-commit-id: 4a4d33047bfbbbbdee6bbac159b98ad289787cf3

[33mcommit e5a959def220fb3005fd95d2f7736f00368653ae[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:32:02 2020 -0700

    Added evaluator to DistEnergyTrainer
    
    
    Former-commit-id: 2e0f400a39116392144b274d365789ebe0c62cb7

[33mcommit 7cab0eeaf69c96f1178f5c6c90d68220dc167f77[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:28:52 2020 -0700

    Added evaluator to DistEnergyTrainer
    
    
    Former-commit-id: dc7c78c4946de8745d2106e65b34821c79c48bc6

[33mcommit f608904c373ce15e6bc3e9a5c7245b3e0c257c13[m
Merge: decc5cf 66609d5
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:03:47 2020 -0700

    Merge branch 'master' into amp2
    
    
    Former-commit-id: 95005b1d3c067ea5334a99c6490c723074cc2e7b

[33mcommit decc5cf8f430fbae694524308a893af27e2b33ee[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 12:03:34 2020 -0700

    Config fix
    
    
    Former-commit-id: d0c9e8d13cd0e854d66a0932bbae8fadda001fce

[33mcommit f82e1633f595ea83e21ffef32b12d8658e758564[m
Merge: 4be35a3 66609d5
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Sep 9 07:41:23 2020 -0700

    Merge branch 'master' into amp2
    
    
    Former-commit-id: 4a7b1b3307ff1c8c7767656516ec7268344f2c3e

[33mcommit 66609d526741d651723367fc199e5a84cfaab013[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 9 07:28:01 2020 -0700

    Fixed metric computation bug in DistributedForcesTrainer (#75)
    
    
    
    Former-commit-id: 8a1c4f9cd77532641287f2434abedda795e30a9a

[33mcommit 4be35a322ef27be00ae8182c254a385442f83186[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Sep 8 17:01:11 2020 -0700

    Updates dist energy trainer to use Evaluator
    
    
    Former-commit-id: a63de6b63f9c1569c39ed1cb3618a101f6a3072f

[33mcommit 561ff986fb77ec635d5d91ba95123e89d9d88e72[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Sep 8 03:04:54 2020 -0700

    Pinned memory for forces trainer
    
    
    Former-commit-id: 898ec52fe5b3ec875eaa30916220e3f23ed8df19

[33mcommit 1d220300ac6817549063725e10c861acb66578f2[m
Merge: 7a33f85 ccd7052
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Sep 8 03:02:23 2020 -0700

    Merge branch 'master' into amp2
    
    # Conflicts:
    #       configs/ocp_s2ef/base.yml
    #       ocpmodels/models/dimenet.py
    #       ocpmodels/trainers/dist_forces_trainer.py
    #       ocpmodels/trainers/energy_trainer.py
    
    
    Former-commit-id: 28db00f0ba9fdf1bcd3a58c1a2e2f298fd23ac4d

[33mcommit 7a33f85a42fc0cd052926b64792dd79ddb2605b9[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Sep 8 02:48:08 2020 -0700

    Added AMP support to Forces Trainer
    
    
    Former-commit-id: 72603dfa154954668a4c782e339b8dc917f48699

[33mcommit ccd705259e4d4ba933dce991438852dc01face90[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Sep 7 14:47:18 2020 -0700

    Replaces BatchNorm with LayerNorm in CGCNN; reorganizes code a bit (#73)
    
    * Better s2ef cgcnn config
    
    * Replaces BatchNorm with LayerNorm in CGCNN; reorganizes code a bit
    
    Former-commit-id: 493befee6a05d0abc898bde0c1c4a56c1809f881

[33mcommit fed3ebc864d15645d4396520d3b5045a430ec0be[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Sep 7 14:43:43 2020 -0700

    Minor updates to make tests pass + PyG 1.6 compatibility (#72)
    
    
    
    Former-commit-id: cfb7672d2e309b63ef2b857f4e99d89196523abe

[33mcommit 5c3e6f890f5a1bb257ad513212771e3054e88bda[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Wed Sep 2 12:23:19 2020 -0700

    AMP + Forces Trainer
    
    
    Former-commit-id: f5faa4aa65080f463adf76b86c45f404b1b727f8

[33mcommit 68d3bdc2e87b4f28077f46c2b3eed22b5d39407d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Sep 1 20:47:04 2020 -0700

    Version bump to pytorch 1.6, pytorch-geometric 1.6.1 (#71)
    
    
    
    Former-commit-id: 57bf4fdc3022e32915f5a7a7e7c234ff9b73bebf

[33mcommit 2e73c27da77508eadc41bd48dd136cb5a31e447b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Sep 1 13:54:08 2020 -0700

    Adds a separate `Evaluator` class that trainers can import with a wider range of metrics (#67)
    
    * Adds an `Evaluator` class that trainers can import
    
    * Adds metrics for S2EF and IS2RS
    
    * Evaluator tests
    
    * More tests
    
    * Don't use old code
    
    * Updates ForcesTrainer to use Evaluator + some additional cleaning up
    
    * Updates distributed forces trainer
    
    * Adds difference in norms as a metric for forces
    
    * Shortens list of default metrics for s2ef
    
    * resolves bug with dist_trainer
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 3471ef9983bf5bafd9863c5e1168db3f765d8318

[33mcommit f68956acc931335d14152f8f5222cba385b1be3e[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Sep 1 15:07:31 2020 -0500

    adds run_dir flag to main run script (#70)
    
    
    
    Former-commit-id: 68a8e21a1250fd6c4b87d9a3fc07c09dc4e3c950

[33mcommit 81ec9a1081b4c219bc9c4b4585945e7398dc3cbd[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Sep 1 10:52:44 2020 -0700

    AMP Support
    
    
    Former-commit-id: 02037e8a10e90a2c4470ebf3e81bdf6dec650bf4

[33mcommit 733593c719f6429ff278d3fe8981080eecb894e5[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Tue Sep 1 10:51:25 2020 -0700

    AMP Support
    
    
    Former-commit-id: f763d08ca19163fb59b077358f1a705590e1b992

[33mcommit a02980d5013ece67a10e62542f25ab8539396afd[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Aug 31 14:18:16 2020 -0500

    store checkpoints at eval_every step (#68)
    
    * store checkpoints at eval_every step
    
    * Updates checkpointing logic for forces trainer
    
    * Corrects epoch count when checkpointing
    
    * Gets back eval_every != -1 check
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: b886b638f496473eb75be57eb3852fdc812df9dd

[33mcommit f2b80cbd4c3137b1ccd8cc8a0f71586d8caad6ae[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Aug 28 17:50:20 2020 -0500

    adds max neigh filter to pbc_distance() (#66)
    
    * adds max neigh filter to pbc_distance()
    
    * applys filter to distance_vec and offsets
    
    * Separates out neighbor list pruning into a function of its own
    
    * Removes unused param
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: e4b262110056cea97d0cd6abf41ca09a288064a0

[33mcommit 7bf30c92763729997addb6c325d709a4a4f8d2f7[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Aug 28 17:15:42 2020 -0500

    eval_every for ddp trainer (#65)
    
    
    
    Former-commit-id: 8c894e4584f8c07c2ac5f07d4d5695d3630a97a5

[33mcommit 4cf10afd30fd13cf665c6922c6bc7253188d23f1[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Aug 28 14:57:35 2020 -0500

    resolves ml_relax bug caused by recent changes (#63)
    
    
    
    Former-commit-id: ab1278de84d8467a76be92ddd182a2cf56f6dfb6

[33mcommit 0716ae91cb7f7e55be5e8dfb6180da98ca7fd9c8[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Aug 28 14:51:28 2020 -0500

    evaluate model every N parameter updates rather than epochs (#62)
    
    * evaluate model every N parameter updates rather than epochs
    
    * modified to eval_every
    
    * typo
    
    Former-commit-id: 10a1a7d9e51cade19c7b89e53069c7a33c72ac25

[33mcommit c0f98e4a55ef8a4b457c3539bc2254073b6d0011[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Aug 28 14:04:47 2020 -0500

    updates configs for s2ef task (#61)
    
    
    
    Former-commit-id: 905bad48143839a3fc3cc2d0f50c9b9d8872600c

[33mcommit b2eca848ad80bfd7641a9ca79591e00e3eef5947[m
Merge: 0a917cc 70e2b8c
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 28 10:44:21 2020 -0700

    Merge pull request #48 from Open-Catalyst-Project/ddp2
    
    DDP implementation for the forces trainer.
    
    Former-commit-id: b33838f0f0ee6dbdc99e5217d1c9ab1ba4146370

[33mcommit 70e2b8cd0a320da41c2ceed1d1616195654da095[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 28 04:05:33 2020 -0700

    Configs
    
    
    Former-commit-id: a51b3c08fff620231895f58ff0e73eb9f6ecfd70

[33mcommit 70801ca82e52b9c8355f15a836d053f4efd07a2a[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 28 04:03:49 2020 -0700

    Distributed LMDB dataset
    
    
    Former-commit-id: d075b921599c0b775c3f925eba8afed0193c8fb4

[33mcommit bd5f941337bef5eff94f6fce1ba8115977d74665[m
Merge: 6e01c28 0a917cc
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 28 01:13:37 2020 -0700

    Merge branch 'master' into ddp2
    
    
    Former-commit-id: 0496ba10080c8400ee7588f00c07064a1d2f3cc9

[33mcommit 6e01c2860bc6e309224f35bbbae43377ef3784ec[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 28 01:12:56 2020 -0700

    LMDB dataset
    
    
    Former-commit-id: 804d07a69d755ff96e351bc7f541bac385fddf38

[33mcommit 0a917cc93cfb373df06d89ac2332dff3870e6071[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 27 17:07:38 2020 -0700

    Stores length of each LMDB as a separate entry; speeds up dataset initialization (#60)
    
    * Minor dataloading speed improvement: Lazy-encode keys to ascii when getting specific objects
    
    * Read length from lmdb file directly
    
    * removes sampler logic from dataset
    
    * Saves count within each lmdb during preprocessign
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 841bcd6a54838ec70e0512b2c8e11aed6fc798b5

[33mcommit 3b336ababab91ab7af6a3aedaf1bbfcf7b6efde9[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 27 13:02:04 2020 -0700

    Saves checkpoints at best val accuracy only (#57)
    
    
    
    Former-commit-id: caff74436dc1301ac0a6a3a16db4ad55f76a3eda

[33mcommit df6201b313fea40b9b97bebae52801e38de73db5[m
Merge: dbcdad7 c49da31
Author: anuroopsriram <anuroops@fb.com>
Date:   Thu Aug 27 11:54:41 2020 -0700

    Merge branch 'master' into ddp2
    
    # Conflicts:
    #       ocpmodels/datasets/trajectory_lmdb.py
    #       ocpmodels/models/dimenet.py
    
    
    Former-commit-id: e2759a0d0cdb1ed853e116dad00b59020db5593a

[33mcommit c49da31e9e976c62b805ddcd96df58adee76bdc2[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Aug 27 12:34:22 2020 -0500

    Forces dp bugfix (#59)
    
    * resolves recent dp bug
    
    * batch.fixed to main device
    
    Former-commit-id: 33afda36711e6d0ffe4208982453edaa1056b07c

[33mcommit dc15d485349b3fc4afb7383659d1d23c50bfdbba[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Aug 26 22:44:30 2020 -0700

    Removes TrajSampler import from datasets
    
    
    Former-commit-id: c5a723def840bbd6561e1d4773ca56f624d50b56

[33mcommit c4125c09cc3d88bb7fe0e2c2c0612beae433d30f[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Aug 26 23:52:00 2020 -0500

    removes traj sampler (#58)
    
    
    
    Former-commit-id: a3b23babcdb5a22b2451a88cca8d98bacd1c2853

[33mcommit ef8d5b4c6524a26ed282061a167d584854966292[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Aug 26 20:03:59 2020 -0500

    Updated preprocess scripts (#53)
    
    * includes updated preprocess scripts for energy/forces
    
    * add try/except
    
    * Updates preprocessing script for initial to relaxed tasks
    
    * updated s2ef preprocessing scripts
    
    * resolves lmdb memory limit bug when #lmdb>60
    
    * typo
    
    * Saves a log of trajectories that are filtered out
    
    * adds script to generate image text file for S2EF task w/ filter
    
    * Adds a check for zero neighbors when preprocessing atoms
    
    * add check for zero neighbors in s2ef preprocessing
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 5e48663f3ea15b8ba1ff681009729e5dea3d13ca

[33mcommit 132d1cca26576a8b874b4c838836a217aef99624[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Aug 26 17:21:55 2020 -0700

    Transfers entire batch object to corresponding GPU than specific attributes (#56)
    
    
    
    Former-commit-id: dc383d723c323b0413e58bc9500dce59765419f1

[33mcommit d5539104e2146ad61068f71c2931375829b76c19[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Aug 26 18:56:44 2020 -0500

    PBC QOL (#55)
    
    * resolved >60 lmdb memory issue
    
    * return distance vec flag, resolved .nonzero() warning on torch 1.6
    
    * cleans up return statements
    
    Former-commit-id: bfd7e9b3fdba6a65a8fdb0f301f2f4e382bd6f27

[33mcommit 14e60e7d997785875b6c206bcfc0514a6d87530c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Aug 25 13:35:15 2020 -0700

    Sorts neighbors in increasing order of distance before pruning (#54)
    
    
    
    Former-commit-id: 68ec19589642725a10064385a46ae96cc50ce14c

[33mcommit dbcdad709d32b79a338d2e4cada8ad6598556155[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Mon Aug 24 23:12:51 2020 -0700

    Commit
    
    
    Former-commit-id: ae98f91dacbd357259b9f6a3c3c17e75d2e61da6

[33mcommit 3d79fac9c6c9e182740be001ec6cb9929e3ac102[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 20 16:33:45 2020 -0700

    Adds parameter for subsampling angles in DimeNet (#51)
    
    
    
    Former-commit-id: 2fabb63c0c00a3618419c7af27ab35199fe01a12

[33mcommit dfc5cb03e8af56fbecec4ae8281bb30516978c66[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Aug 20 16:29:24 2020 -0700

    Removes cutoff radius parameter from pbc since we no longer use it (#52)
    
    
    
    Former-commit-id: af4dbce86e502a9381d4bd1cab102cdbb78dcc63

[33mcommit 07e674850d66fdb0a7f6d2b1ba9df81197a65948[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Aug 18 12:23:59 2020 -0700

    Fixes offset bug when computing angles in DimeNet with PBC (#49)
    
    * Fixes offset bug when computing angles in dimenet with pbc
    
    * Updates tests to use data_list_collater instead of PyG Batch.from_data_list
    
    * consistency with radius_graph
    
    * resolve failing tests
    
    * Remove edge_index ordering in dimenet; it is updated in preprocessing
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: 7f987efcfb76333f0a2e11faee6a2898d458ee51

[33mcommit c926c7bb07d79d8bc97ab44c21f55ad432bf0ad4[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Aug 17 10:07:27 2020 -0700

    Preprocessing, dataloader, model configs for initial structure to final energy prediction (#39)
    
    * Preprocessing script for relaxed energy prediction
    
    * Dataloader, configs, trainer for IS2RE
    
    * Best hyperparameters for IS2RE on 10k
    
    * Filters out systems with large relaxed energies
    
    * Best configs for 10k, 50k
    
    * Typo
    
    * ase.io.trajectory.Trajectory is faster than ase.io.read
    
    * Some more configs
    
    * Removes dummy indices from preprocessing script
    
    * Dimenet scatter dimension bugfix
    
    * Updated configs for OCP IS2RE
    
    * Changes natoms.sum() to pos.size(0)
    
    * Updates preprocessing to be consistent for IS2RE and IS2RS
    
    Former-commit-id: aee27fc8645af9b63d724a632f70702f73505ea6

[33mcommit 222b7aa2018ee80d16a987649081601139d1faa8[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 14 10:31:12 2020 -0700

    Minor config changes for Dimenet
    
    
    Former-commit-id: 3fde9487221151d42fd2282273deae45a1c8b9a4

[33mcommit 551c1ac16929e71cdf7b1654a64ca1d5c4d561f8[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 14 08:49:43 2020 -0700

    DDP implementation
    
    
    Former-commit-id: 6319d349de86125740a958bef68e6e6351546d01

[33mcommit a95427ae7c131821c64f17e0a857ea5c05af498b[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 14 03:00:13 2020 -0700

    DDP implementation
    
    
    Former-commit-id: b41b1966f484cc79d5b964460991a8763919950d

[33mcommit 257b5155b972dab210f10eac8d204b3cba679db2[m
Author: anuroopsriram <anuroops@fb.com>
Date:   Fri Aug 14 01:50:11 2020 -0700

    DDP implementation
    
    
    Former-commit-id: 97cd9f085fd3fd53a01e41d3be2e112df7d3f559

[33mcommit 11bce89910eca48ed967d4bc2feb4e7e151c29bb[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Aug 13 19:55:52 2020 -0500

    preprocessing refactor: removes dummy indices, resolves pbc distance bug, cleans up pbc/dataloader utils. (#47)
    
    * removes dummy indices, resolves pymatgen "bug", cleans up dataloader
    
    * removes extra bits
    
    * updates a2g tests
    
    * includes max neighbor cutoff, resolves compatibility with no preprocessing
    
    * includes latest/updated preprocessing script. adds redundancy to remove zero distances
    
    * updates model tests
    
    Former-commit-id: 3b9eb206b5d9fd5f8364077ea194ebb6443462ec

[33mcommit 8f2adcfcc20c147a705ec0a960b59dfa494047a6[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Aug 12 15:11:58 2020 -0700

    hotfix: adds batch neighbors attribute to data_parallel call
    
    Former-commit-id: c6b357bc65e9972312b303afdb74f3fce6699d10

[33mcommit e5523659f7d8cac49d2730fb68910ce5b8b1c7e8[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Aug 12 14:36:26 2020 -0500

    Pbc max neighbor limit (#45)
    
    * Manually filters out pymatgen edges with large distances
    
    * removes fixed neighbor count in pbc calculation
    
    * updates models for pbc changes
    
    * nonzero overload deprecation pytorch 1.6
    
    * Removes duplicated lines
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 6507794cacabaf4be93baa448edea35c246cb856

[33mcommit 7019be4ff576582055202d310fdf8cc4288703ed[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Aug 11 23:21:34 2020 -0700

    Manually filters out pymatgen edges with large distances (#43)
    
    
    
    Former-commit-id: 97af3dc57c64c18e2ecc8e28842166ff89290a22

[33mcommit f01a9727285a26509658bac4b045a92dafd6bfc7[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Aug 11 12:54:35 2020 -0500

    resolves pbc bug caused by PyG batching (#42)
    
    * resolves pbc bug caused by PyG batching
    
    * remove reliance on max_neighbors
    
    Former-commit-id: c52aa8d8cb8309b1a0dab6a2b9f48d96109d9ac4

[33mcommit aa23173816c55bebc43e0a0a0a9c8a8ae38c5014[m
Author: anuroopsriram <anuroop.sriram@gmail.com>
Date:   Fri Aug 7 00:26:35 2020 -0700

    Added submitit support to run on slurm cluster (#41)
    
    * Added submitit support
    
    * Added submitit support
    
    * Added support for parameter sweeps
    
    * Running through black, flake8, isort
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 4e034ea878930706c57a65e33b79b24ec22ff432

[33mcommit 486b0fa83197a8018700ece72b667f6e572fa4ab[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Aug 5 15:15:19 2020 -0700

    Fixes rotation invariance tests that broke due to (unrotated) pbc cell offsets
    
    
    Former-commit-id: b4dde78e4778d2d8984bc3869e1f5f50164e7558

[33mcommit 76e7bbb3813f96b6565b1b1bcd16917b40ba6a5a[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Aug 5 16:59:57 2020 -0500

    model pbc fix (#40)
    
    * cgcnn pbc fix
    
    * pbc distance fix
    
    * Moves pbc distances to a single unified function
    
    * Fixes dimenet to use the correct distances with pbc
    
    * Adds a test to compare distances
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 0515191436dded47a2f3d98514237c5d56185565

[33mcommit dd8097c69754587a85d0626290d46cdefd79895e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Aug 1 22:51:58 2020 -0700

    Support for random rotations and a few tests for rotational invariance (#34)
    
    * Rotation transform and a few tests for rotational invariance
    
    * Adds pytest to dependencies
    
    Former-commit-id: fbda8a3175f8ca94647b5d5712aedb889bfe0990

[33mcommit 48370e5527b67594508464171813173aebcfa603[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Fri Jul 31 15:09:17 2020 -0700

    adds edge_index to data parallel for pbc calculation
    
    
    Former-commit-id: ed7745f7f8defa1c7b2b695b330ac23811724f4d

[33mcommit 2e809a2c52baf296da33e65bd22b12b1783fdf02[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jul 31 17:06:09 2020 -0500

    pbc fix to schnet, dimenet, cgcnn (#38)
    
    * cleans up energy-only training
    
    * adds schnet_pbc model, corrects edge_index indexing
    
    * combines schnet+pbc modell to schnet with use_pbc flag
    
    * updated dimenet, cgcnn for pbc
    
    * remove extra imports
    
    Former-commit-id: e736ed014185b4bd8d9e6a643806c48a86691026

[33mcommit e6cb9865e705a43acc2c6551cb7e8502dec91fc3[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Thu Jul 30 10:25:05 2020 -0500

    resolves tagging index bug that is caused from the inconsistency in pre-vasp and post-vasp atoms object indices (#35)
    
    
    
    Former-commit-id: 07e18cf70fa7ca51f2641d150f1a326422f14a83

[33mcommit c06a2f99cd623229c2a1c540c8686c75ce6a536d[m
Merge: 6cc3c48 3e26ed6
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 14:49:05 2020 -0700

    Merge branch 'batch_loader'
    
    
    Former-commit-id: c2a36f13528749e32c65cb1f254cdf1cb157136c

[33mcommit 3e26ed688ee647eb73483506074c2a02e49684c4[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 14:46:54 2020 -0700

    append fix
    
    
    Former-commit-id: 07991587fc9a8d4e69686a4dcc3791d202880189

[33mcommit 6cc3c487affdd3afae220c97ced243fa54c9075d[m
Merge: c718db6 a44ae08
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jul 29 16:42:00 2020 -0500

    Merge pull request #33 from Open-Catalyst-Project/batch_loader
    
    Trajectory sampler batch loader
    
    Former-commit-id: 54ed3f347c7e53a5f69fb65f9ed998122642a8e7

[33mcommit a44ae08809c64b2020104080352c8fb8e2e3e7a4[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 14:40:48 2020 -0700

    fix traj idx logging
    
    
    Former-commit-id: 0a727f7370e5b285ecc2b373a962011f6444c1c2

[33mcommit 31b6eed2d32ea888439429708545b5d64c28802d[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 13:57:35 2020 -0700

    inconsistency with do in .lmdb and logged .txt
    
    
    Former-commit-id: 02ec0c77a60959cbc139a9cd98eb6e9085d17717

[33mcommit eaf73597abee4b779a3bff99a7abe4b306517ac1[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 13:40:32 2020 -0700

    bug fix: sampler->batch_sampler
    
    
    Former-commit-id: 23edb0f750c1a3e8cbe209afed76b362ba436be1

[33mcommit 9ad430b74893f3887b95424e515734441e6e315b[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 13:10:30 2020 -0700

    modify config for traj_per_batch
    
    
    Former-commit-id: 399fcc339dece60a0ffc7821deb0842c17097358

[33mcommit 5f4b74d627dc65f69988a6ccb0303e0bb3f1389c[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 12:47:10 2020 -0700

    description
    
    
    Former-commit-id: cbfd42bc4514bab180df2c131db3d2f2514837f9

[33mcommit 364e14949ea2e2c5b08420d432af13aed0d557e3[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 29 12:37:31 2020 -0700

    Adds trajectory batch sampler
    
    
    Former-commit-id: 319232ff663c2b83ac05da338dd598a0e8a2ebc9

[33mcommit c718db6d39b8853366aa0a5fe67481f72601c9e5[m
Merge: 4a6d448 96d640d
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jul 28 18:47:54 2020 -0500

    Merge pull request #32 from Open-Catalyst-Project/ads_idx_preprocess
    
    adds adsorbate flags to preprocessing
    
    Former-commit-id: 7b379feaa96f49da2c1d4e086442a47a080931e0

[33mcommit 96d640d432d05b5ec8c59288c097076d11cab2f9[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 28 16:45:32 2020 -0700

    retrieve original tags from input generation
    
    
    Former-commit-id: 68fa69feb100b19c31f1847643835df2ba5c8d67

[33mcommit a4d97ded8036e0e276c12e411915ceb3e1093124[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 28 13:05:32 2020 -0700

    adds adsorbate flags to preprocessing
    
    
    Former-commit-id: cf32f153cb0ed2db8d029493609aeda5c0c3e551

[33mcommit 4a6d448d8914d814e80e31d22f066ce446c6eecf[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Thu Jul 23 07:18:46 2020 -0700

    adds natoms to data_parallel .to() call
    
    
    Former-commit-id: 46e1dc8912c08070b4f7c50fe04ef308f041002c

[33mcommit c37e462ad7f3b98c55f7bd7a0d8de71f2116033b[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Thu Jul 23 07:05:29 2020 -0700

    add cel information to .to(tensor) in data parallel
    
    
    Former-commit-id: 57c2de88f8586ffbbdf00485be2f380973b8fd56

[33mcommit 9cbd2d9e591109a35c968c9262538e6c5bdfb2bb[m
Merge: c21cbb2 66b6c04
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jul 22 12:20:32 2020 -0500

    Merge pull request #31 from Open-Catalyst-Project/data_parallel
    
    mutli-gpu training/DP initial implementation
    
    Former-commit-id: f1a685ac95365319b4db9e8a9457c6e8d4848714

[33mcommit 66b6c04c4f24cebda26c9d4d05919ae5faa09d79[m
Merge: 69f8f79 c21cbb2
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 22 10:13:14 2020 -0700

    Merge branch 'master' into data_parallel
    
    
    Former-commit-id: 4553545bda93d096299c62b592fb445bc4d3566d

[33mcommit 69f8f79cceebbc1f3d4bac7b8f9015577d05513a[m
Merge: 3bf42b8 7296e69
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 22 10:05:57 2020 -0700

    Merge branch 'data_parallel' of github.com:Open-Catalyst-Project/baselines into data_parallel
    
    
    Former-commit-id: ba61f3059365cf3a2252b4cf5bef24b7012c13a6

[33mcommit 3bf42b8fe5924640341276924461935f3d3f0a27[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 22 10:05:41 2020 -0700

    data parallel compatability with TrajectoryDataset
    
    
    Former-commit-id: 2b24c2e18064102e7ee3d89b182623822542a0f0

[33mcommit 7bd8d4db333c038474ffdd57c79f2c3a00297272[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 22 10:04:13 2020 -0700

    compatability with TrajectoryDataset. predict() call fix
    
    
    Former-commit-id: 8053c98b4b1277b2609e610637252f877d17c11c

[33mcommit 7a88eab5bac1cec47b0ba8b446d6007a886995eb[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 22 10:03:30 2020 -0700

    store fixed indices by default
    
    
    Former-commit-id: 76d3e0e464b72d600044d78513f6aa3de8941021

[33mcommit c21cbb29122e6bc408cff99462597caa91da8988[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jul 22 12:36:22 2020 -0400

    Updates CGCNN and DimeNet to work with new dataloader and ForcesTrainer (#30)
    
    * Better trailing slash handling; show error if no LMDBs found
    
    * Support for training / evaluating on only free or all atoms
    
    * Tidies up free / fixed masking a bit; adds updated config yamls
    
    * Updates CGCNN to work with ForcesTrainer
    
    * DimeNet updates for energy + force training
    
    Former-commit-id: 4272a36e861cbd8d138dd4a7fa31898818cf72ca

[33mcommit 7296e69d0ded10b500266cacd920f1e3cbec886f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jul 22 09:00:10 2020 -0700

    Free cached GPU memory after every training epoch
    
    
    Former-commit-id: 8fd1027c51e65e727f48da2c0195b9ddb2b00bb2

[33mcommit 18c4fd96fea9bbbbd740e344adb25a1d88772baf[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 21 16:20:27 2020 -0700

    fix batch.to(device) on validate call
    
    
    Former-commit-id: 886eaab260f30acf701fc8a0b232d206c5f0c915

[33mcommit 03d4caa86208ef3223c9e950cf0799c12a239344[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 21 11:12:33 2020 -0700

    added force computation to remaining models
    
    
    Former-commit-id: b8ae98140b0709fb4098dddcbbf1eb7c9751515e

[33mcommit 414612c73d20b14c31032c977c446b37a40e9af1[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 21 11:02:28 2020 -0700

    inherit load_model() rather than redfine
    
    
    Former-commit-id: d1e4e6b6acaa3c22e3368f74a76acbcef995c336

[33mcommit f6bbe9492026a1fd333dc5139fd6e831d7be2d61[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 21 10:54:48 2020 -0700

    move _forward, _compute_loss, and _load model to forces trainer, backwards compatible for other models, forcetraining->regress_forces
    
    
    Former-commit-id: bf175a8dd4a4aaed03d39ec77a3057d7aa965d72

[33mcommit d749315b4514743af7397599131e3dde4af14bad[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 20 18:02:49 2020 -0700

    mutli-gpu training/DP initial implementation
    
    
    Former-commit-id: 94f2a0fd4df66bec736d83b70e88ab56b6173727

[33mcommit 6bce5774a010255b4b663f432af98c83c40b303a[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jul 20 12:57:05 2020 -0400

    Adds support for training / evaluation on free / all atoms (#29)
    
    * Better trailing slash handling; show error if no LMDBs found
    
    * Support for training / evaluating on only free or all atoms
    
    * Tidies up free / fixed masking a bit; adds updated config yamls
    
    Former-commit-id: 54ea8dc65620ba1db6437908981886d09b48b72f

[33mcommit 03fd272bc54d669e4fc9f7ec2ed4113708ed0f92[m
Merge: 5f30737 bb51784
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Sun Jul 19 15:18:03 2020 -0500

    Merge pull request #27 from Open-Catalyst-Project/pbc-neighbor-offset
    
    Neighbor offsets for periodic atoms + adslab reference energies
    
    Former-commit-id: 782879b08ff68cff159614a9611ba7b10275327b

[33mcommit bb5178415d75cb9d0c897c98c9fe368e13e4eff4[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Jul 18 11:31:41 2020 -0700

    Makes tests pass
    
    
    Former-commit-id: c5c524b831292bf59ba530af438408d553c9491f

[33mcommit ed00c27821f024a1ebcbda708bd9372938ba00cc[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Jul 18 11:04:03 2020 -0700

    Reverting some changes unrelated to preprocessing
    
    
    Former-commit-id: bcaa870eea88fbc682c84b5643728d2d04c15df2

[33mcommit 0fff435f0f35ec35d678be1e2a57bb9dd6744ff0[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Sat Jul 18 09:33:55 2020 -0700

    round2 preprocessing script (adslab+cell info)
    
    
    Former-commit-id: 16c5a250df4aed31b7e2e2bd5525323b92299804

[33mcommit dc56143412fde0411e78a76ac2078abb00a63c1d[m
Merge: 7dae3fc 5f30737
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Fri Jul 17 13:53:09 2020 -0700

    Merge branch 'master' into pbc-neighbor-offset
    
    
    Former-commit-id: aee773020de4451dba9159a763dbb7b5b1d311a4

[33mcommit 7dae3fcd6c4bb616b3d192e9d4c333276b87a44a[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Fri Jul 17 13:22:12 2020 -0700

    added adslab_ref correction parallel processing
    
    
    Former-commit-id: b5416445423b239824227b261388d11c670690b6

[33mcommit 5f30737cd7722874e43e4368d522074bf63cb04a[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jul 17 12:37:07 2020 -0400

    Save trajectory length as well when preprocessing (#28)
    
    
    
    Former-commit-id: 21da4be4b2da8a04ea6fc1d6b30f22224a0c3648

[33mcommit 539ecd56703da4405875e2d4cdfe10e8fec875dc[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Jul 16 19:18:00 2020 -0400

    Opens connections to LMDB only when needed (#26)
    
    * Opens connections to lmdb only when needed
    
    * Minor
    
    Former-commit-id: 69fb815ba887672723922b0ad6c7603d608ad1e6

[33mcommit 8be69015482d84f144d79bf6ebcd46a9a4381c05[m
Merge: 7d39922 94cf459
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Jul 16 15:50:11 2020 -0700

    Merge branch 'master' into pbc-neighbor-offset
    
    
    Former-commit-id: d46f4db7bb81bc5229d4b401c48dd1b355364ba2

[33mcommit 94cf459d3916d881e23240881df8e030d55349b2[m
Merge: 72f165a cfd2a27
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 14 15:40:37 2020 -0700

    Merge branch 'processing'
    
    
    Former-commit-id: ec8e73021298a9bf450c7822e09d3187b51cb881

[33mcommit cfd2a27d3e84df8307abf8916274c7e847f2375a[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 14 15:38:10 2020 -0700

    documentation
    
    
    Former-commit-id: 684948be1d6c6c73315b639788c8ec45cf320d06

[33mcommit 72f165a2ee5522208feb07d4ef3f057bc6fd308a[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jul 14 17:37:25 2020 -0500

    Parallel preprocessing from ASE to LMDB (#25)
    
    * parallel preprocessing
    
    * Gets number of images from args.size, saves sampled ids in txt, plays well with tqdm
    
    * Typo
    
    * Updates dataloader to read from directory of LMDBs
    
    * adds the indices of the fixed atoms to the preprocessing script
    
    * added fixed atoms flag to parallel script
    
    * binary vector instead of explicity indices
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 584e764832e206bf1f2dae1ff20125367b23bdef

[33mcommit 1674a5712238fefa9ea89c440ef02caaf1f4717e[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 14 15:34:08 2020 -0700

    binary vector instead of explicity indices
    
    
    Former-commit-id: ea929522983cd3e42756b727bb9f07d5c8da6816

[33mcommit ecc9530065c3d67ab2fc9085fab6197fb5cc6a22[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 14 14:51:41 2020 -0700

    added fixed atoms flag to parallel script
    
    
    Former-commit-id: 92ba5a976ef11e7d281280d4641d6be850498de5

[33mcommit 6743de6a1caa918b43d2a0966733e58d35bdf9ce[m
Merge: 6af2fe6 dfa90dc
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 14 13:59:33 2020 -0700

    Merge branch 'processing' of github.com:Open-Catalyst-Project/baselines into processing
    
    
    Former-commit-id: e2338135bf198db298ab0c80c80cebf4bbbf07a2

[33mcommit 6af2fe612b2eede1fdc02305cd7f92acfdd58753[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Tue Jul 14 13:58:59 2020 -0700

    adds the indices of the fixed atoms to the preprocessing script
    
    
    Former-commit-id: cfb4344a67495e593e0f7aee59c73cdf0e143ec5

[33mcommit dfa90dc0f481d57cbbfc305ca04a389848c8f9aa[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jul 13 17:12:52 2020 -0700

    Updates dataloader to read from directory of LMDBs
    
    
    Former-commit-id: 2c483beaee5cf4923aa4d58d6a5a9a6aaf1c7bd0

[33mcommit eade587329a2892a6c2e2b638d41661b255e68cc[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jul 13 16:25:17 2020 -0700

    Typo
    
    
    Former-commit-id: e25748260b5df760cedd01837c3a5702dacafcdd

[33mcommit af0768e7267ef7eab0ad92bc156139c6288e1080[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jul 13 16:16:55 2020 -0700

    Gets number of images from args.size, saves sampled ids in txt, plays well with tqdm
    
    
    Former-commit-id: 66c6b590378d78658122fc60fc6e1a3c9e6822ee

[33mcommit 581fcd505379dee85fd522d8a2e60b8c6abc2760[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 13 14:53:37 2020 -0700

    parallel preprocessing
    
    
    Former-commit-id: 72184fa587659cea827015089c7e4d2c678b861d

[33mcommit 7d39922716edb76209350e41353ac701e79e3907[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jul 13 14:25:31 2020 -0700

    Preprocessor returns offsets for pbc
    
    
    Former-commit-id: b204a529ed46925c65bb562207a4a96ee4ce95c5

[33mcommit da075e0211c7a5b3d44ced4ea84f77188513c560[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 13 12:06:35 2020 -0700

    updated preprocessing script
    
    
    Former-commit-id: 80badadab147082906cee1e6c711fd901ca78d07

[33mcommit 6d34b6ea4730f39bc541fbfdb794523a05d55a00[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 8 16:41:21 2020 -0700

    remove unnecessary bits
    
    
    Former-commit-id: e9296d830369ce34d2166c5bf3f44413efaae72b

[33mcommit 11d1bea492d9188a036fef1f341fe29e3c2bb139[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Wed Jul 8 14:52:42 2020 -0700

    added catch for images with no calculator attached/error in processing, preventing script from crashing when encountered
    
    
    Former-commit-id: 5c0f136a5e4a90d3b9f82a8bd9e9d06cec9c2a1a

[33mcommit 8a7747b4b7c7932c0589ba607a9183354c63b855[m
Author: Weihua Hu <weihua916@gmail.com>
Date:   Tue Jul 7 14:20:09 2020 -0700

    add transform to trajectory dataset object (#24)
    
    * add transform
    
    * fix style
    
    Former-commit-id: 55b6373d0b0629975b1f3c125aa68be77e5622c2

[33mcommit 9ec4ac5744b7c7dc8bca1709c659e1b8ff3fa7cd[m
Merge: ec05253 83a00d8
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Mon Jul 6 14:38:56 2020 -0500

    Merge pull request #23 from Open-Catalyst-Project/predict_fix
    
    Resolves predict() bug cause form earlier commits
    
    Former-commit-id: b670250095e7a2748929f711d7a1ec1353510af1

[33mcommit 83a00d8a5414a02d495ded37094e29210a5e65ae[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 6 12:01:34 2020 -0700

    Update TrajectoryDataset to use new processing
    
    
    Former-commit-id: 7cd056cedbe2b0319e8b4a769e4b6680327ff826

[33mcommit d61736bda9112cda952069d886b208fdd7c4908e[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 6 10:51:50 2020 -0700

    remove old atoms_to_batch preprocessing code
    
    
    Former-commit-id: f62f794ec256fe95fdd8aaabb2702e66b9ae0044

[33mcommit 67a38ac05887d82d368263624fa8063ac89604d8[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 6 10:49:53 2020 -0700

    moving away from obsolete data preprocessing in TrajectoryDatset
    
    
    Former-commit-id: 52fffa98f287f2e7f783299bf655dcfb1ae58109

[33mcommit adced4ce7b8eec3be50d11bf6e23ba9f892942d5[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 6 09:14:40 2020 -0700

    remove unnecessary import
    
    
    Former-commit-id: 310fd4e6480e14fcecb9259a6d759bd85ebcec3c

[33mcommit 75e6e4cf879b8c6b6f5f3b055ef310a042a53526[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jul 6 09:10:29 2020 -0700

    resolves predict() and ml relaxation evaluator bugs caused from trajectory_lmdb dataset commit
    
    
    Former-commit-id: 27a7b11f22c5b11ff05d5923de8a578fc6d31f16

[33mcommit ec0525345f2245b9880c8987b90ebc44ef3ff972[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jul 1 11:53:26 2020 -0400

    LMDB dataset / dataloader for loading multiple trajectories (#21)
    
    * Adds helper functions for imports, config; main.py works for energy / force models
    
    * Configs for force models
    
    * Updates older configs to play well with the default trainer
    
    * Reverts default logger to tensorboard
    
    * Includes mmf license
    
    * Initial version of trajectory folder dataset
    
    * Trajectory subsampling logic
    
    * Hacky working version of larger-scale trajectory folder runs
    
    * Ports over relevant portion of Brandon's preprocessor for now
    
    * Updates normalizer to be backward compatible
    
    * Starting to train on the 1k set
    
    * Configs for training SchNet on OCP-1k
    
    * Gets back create_graph
    
    * Normalization params for new data snapshot after removing outliers
    
    * Shifts to the new `AtomsToGraph` preprocessing
    
    * Backward-compatibility and reverting unnecessary changes
    
    * Removes unused imports and driver script
    
    * Incorporates Muhammed's comments
    
    * LMDB dataloader for trajectories
    
    * Use a larger batch size for val as well
    
    * Correction in comment
    
    * Smaller batch size of 64 for OCP-1k eval
    
    * Adds lmdb to requirements
    
    * resolved logic bug when normalize_labels=False
    
    * Reduces batch size, adjusts lr warmup / max_epochs
    
    Co-authored-by: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
    Former-commit-id: c0b8e3f6d6d9a7a14f6c68584294eaacbee5b587

[33mcommit 49b713569a65a54815c77011171f577141dd2bd2[m
Merge: db5ff2f 651611e
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Thu Jun 25 15:50:10 2020 -0700

    Merge pull request #20 from Open-Catalyst-Project/preprocess_and_testing
    
    refactored preprocessing from atoms to graphs and added tests
    
    Former-commit-id: a982a1569961bf7043ddaed0e915fe8dc964c9da

[33mcommit 651611e57f1e81bf8abd20e327ddc2d8c04b626f[m
Author: wood-b <b.wood@berkeley.edu>
Date:   Wed Jun 24 21:47:47 2020 -0700

    refactored atoms_to_graphs and testing based on PR comments
    
    
    Former-commit-id: ad786186f3cb98293df8aa7ef14f07132524a3ae

[33mcommit db5ff2f4bafdd878792c02861128608c29579831[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jun 22 09:59:00 2020 -0700

    clean up arguments for validate_relaxation
    
    
    Former-commit-id: 5f899b2549f37b2f72f0d974be7aca194b7982fd

[33mcommit 562a048b72b55573c57819655a840a3e5d77f0e5[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri Jun 19 16:55:46 2020 -0500

    ML-relaxation evaluation (#22)
    
    * Renamed "MDTrainer" to "ForcesTrainer" and "COCuMD" dataset to "TrajectoryDataset". Adding ML-relaxation evaluation metric.
    
    * ase_calc load error check
    
    * removed hard-coded evaluation settings, updated task config
    
    * DEBUG: initial parallel scheme
    
    * removed parallel evaluator for time being, direct ml results to "results_dir", clean up forces_trainer
    
    * fix directory to be consistent with other examples
    
    * pre-commit hook
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 3bac8f7923012576cf53132be6e8c0babdaf8cd7

[33mcommit 41b7dabf505321b8505633131ffd4e93246febd8[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jun 19 13:54:58 2020 -0400

    Adds auto-import helper function + config yamls for force models (for larger training sweeps) (#19)
    
    * Adds helper functions for imports, config; main.py works for energy / force models
    
    * Configs for force models
    
    * Updates older configs to play well with the default trainer
    
    * Reverts default logger to tensorboard
    
    * Includes mmf license
    
    * Removed hardcoded `mae`; reads metric from config
    
    Former-commit-id: c933700eb28f30e5d6486578d38cc4fdd5f62ee6

[33mcommit 81a462948e316a1c5a4e6e356782bd2abf359a9c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jun 17 16:45:28 2020 -0700

    Bugfix in bond feature dimension for example training script
    
    
    Former-commit-id: d1a92b694f6dc953c642294985571da2002e8dbe

[33mcommit fc60795eccffe81b5d2c852cc71efd1fbc5889a0[m
Author: wood-b <b.wood@berkeley.edu>
Date:   Wed Jun 17 02:45:54 2020 -0700

    refactored preprocessing from atoms to graphs and added tests
    
    
    Former-commit-id: 4ff827292fc04b754cc004571ea4ae875a787241

[33mcommit 89ee914c8a84c0b2b931dd9fd280280736f4d216[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jun 16 20:57:33 2020 -0700

    Fixes bug with scripts/train_example.py
    
    
    Former-commit-id: a694731177b16323554ef7bc2be6cea2605300dc

[33mcommit e979d72df3426a7c4fbd53b212170b33ee36bd63[m
Author: Muhammed Shuaibi <mshuaibi@andrew.cmu.edu>
Date:   Mon Jun 15 14:30:43 2020 -0700

    allows wandb to log multiple models being run from the same script
    
    
    Former-commit-id: 117b2b9f5a6f354ec9f064bcf89866a51fd66fd7

[33mcommit e7957d68e1ef3781d62e0ab35b46893b2d3eed1c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Jun 11 14:01:44 2020 -0700

    Removes `grad_input` index config parameters for md runners
    
    
    Former-commit-id: 0e0236b9019ce491bedd066dd91134d41fba891c

[33mcommit f7739f3c0098984ad5304a8262f1960c9d81a4e8[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Jun 11 02:31:51 2020 -0400

    Speeds up inference by converts ase atoms to batches on-the-fly (#18)
    
    * Converts ase atoms to batches without writing traj files during inference
    
    * Removes dummy targets from evaluation batch
    
    Former-commit-id: 08a73f4ede14a2ebdb0cce8ae49cbf9c7dd7f2a8

[33mcommit f4215a6e7c3252461bf1a4868c0a116194029260[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jun 10 16:04:48 2020 -0700

    Adds function for loading pretrained checkpoints
    
    
    Former-commit-id: f97eddfcf711b4964871d7f9bec8bdf2e975a1ab

[33mcommit f3c2395b0c39f54310cb8c878924951ba065d7c1[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jun 10 16:50:33 2020 -0400

    Imports DimeNet from PyG to our codebase (#17)
    
    * Implements DimeNet from pytorch-geometric
    
    * Adds edge weighting to CGCNNConv
    
    * Propagates num_gaussians from config to everywhere
    
    * Removes indexing for data.x to get positions (and grad_input indexing)
    
    * DimeNet seems to be working quite well; slightly better than SchNet
    
    Former-commit-id: 2e3f3537719238d2b4c43f92dd945b88664442dd

[33mcommit 7b27a4c7695f797e889d473acc3e8a7e905d2dc0[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jun 10 13:56:49 2020 -0400

    Adds edge weighting to CGCNN, removes having to persist grad_input indexing (#15)
    
    * Adds edge weighting to CGCNNConv
    
    * Propagates num_gaussians from config to everywhere
    
    * Removes indexing for data.x to get positions (and grad_input indexing)
    
    * Changes `radius` to `cutoff`, defaults `grad_input_mult` to -1
    
    Former-commit-id: 5acaa95360be23b8d35eb65538edc766f3b868bd

[33mcommit b553bb3c16cbf4edde7c96a789aca2807dda5615[m
Merge: d988ce6 93d1687
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Wed Jun 10 12:50:37 2020 -0500

    Merge pull request #16 from Open-Catalyst-Project/schnet
    
    Imports SchNet from original implementation (requires PyG and Pytorch version bump to 1.5)
    
    Former-commit-id: d086cc7eaedbac2966fcc0333e4a931200f039a8

[33mcommit 93d1687332368bef4faacbd59f2155490fd059e7[m
Merge: 0cf02e1 d988ce6
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jun 9 23:27:48 2020 -0700

    Merge branch 'master' into schnet
    
    
    Former-commit-id: ce1829691e944a07e53271baf87eb8083472a6c1

[33mcommit d988ce685ad9c4ef495793f870e5bc284ba0414a[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jun 9 23:26:03 2020 -0700

    Adds `shuffle` flag to COCuMD dataset to shuffle batches per epoch during training
    
    
    Former-commit-id: 7627b8fc0e93c6b335f4a1b8ddf4aa6f93326426

[33mcommit fbf5c25f396b8de4ae308d26c06e61c96110128b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jun 9 23:19:23 2020 -0700

    Adds config support for wandb project name
    
    
    Former-commit-id: a09c1e83a27136d5dd1789de6aa6fd41ee8573f6

[33mcommit 0cf02e15ab6fbb9143fdb27287580f1ecfd44c83[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jun 8 23:15:09 2020 -0700

    Shifts to SchNet from pytorch-geometric (requires upgrading to PyG 1.5)
    
    
    Former-commit-id: efdab599d60dc82bfefaf05229b0b6341508e68e

[33mcommit 1a647c0b3fa1d8c2a276fbab8e7d2545e1eedd0a[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jun 8 15:31:05 2020 -0700

    Minor correction in readme: cuda 10.0 --> 10.1
    
    
    Former-commit-id: 5200af6600e17ba702714333f3afe6ddcf6d0c86

[33mcommit 55c642ed4a93d48d4684903f78ba0580849074f7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jun 8 15:26:48 2020 -0700

    Version bump to pytorch 1.5, pytorch-geometric 1.5, cuda 10.1
    
    
    Former-commit-id: 61af5258fe88230e9b38a0cc8285cecaa526cd47

[33mcommit 3a613c72f10353927da979539b17797e0926878e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Jun 4 23:06:27 2020 -0700

    Adds note about `pip install -e .`
    
    
    Former-commit-id: 959d2041261d3ca2103e469c1d79278a883abd25

[33mcommit 121b211f6db616f2a4603ce6d473b64da47440e4[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Jun 5 01:47:19 2020 -0400

    Implements SchNet (#14)
    
    * Implements SchNet
    
    * Removes python path import, adds a note about the default force loss coefficient
    
    Former-commit-id: e028b646011144fe49813f4d2d556dfe7e2b0298

[33mcommit ffdf58bea50ee42e2b24b829b537f740d8a28062[m
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Wed Jun 3 12:30:20 2020 -0400

    Parameterize kernel and covariance matrix for GP (#10)
    
    * Parametrize kernel and covariance matrix for GP
    
    * Propagate parameterized GP through GPyTorchTrainer
    
    * Call ExactGP.forward method directly
    
    When trying to load a previous state and rerunning, for some reason we
    were getting argument errors (e.g., passing 3 positional arguments and
    expected 2) when calling the GP model, which implicitly called the
    `forward` method. This somehow got fixed by calling the `forward` method
    explicitly.
    
    * Add init file to modules to fix importing
    
    * Check for empty ase.db objects
    
    * Pass checkpoint sizes correctly for GPyTorch
    
    * More consistent GP state loading for CFGP
    
    * Fix bug with default cov matrix in CFGP
    
    * Clarify and correct GPpyTorch training procedure
    
    Former-commit-id: 0a50d3b55d5731051b49128e9e722a9fb31c8479

[33mcommit 24dc308f8e0f0410a42bf1ded3f171b13ddbe55d[m
Author: Caleb Ho <caleb.yh.ho@gmail.com>
Date:   Tue Jun 2 22:53:53 2020 -0700

    Create environment files for GPU dependencies (#13)
    
    * Create environment files for GPU dependencies
    
    Common dependencies were split out of `env.cpu.yml` into
    `env.common.yml`. GPU specific dependencies were put into `env.gpu.yml`.
    This allows the user to use `conda-merge` to create a final config which
    can be passed to `conda env create`, e.g.
    ```
    conda-merge env.common.yml env.cpu.yml > env.yml
    conda env create -f env.yml
    ```
    
    * Add missing dependencies required to run scripts/train_example.py
    
    * Removes `conda-merge` help msg and note on extra dependencies since they're already covered
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 2e5a6dc7ae5c98a77f4120912b5b68e5b01d3814

[33mcommit 158ab298488c76a7e663717cf955e46c8f851fe1[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Tue Jun 2 14:49:35 2020 -0500

    OCP-ASE calculator support (#12)
    
    * Adds dataloader for MD trajectories
    
    * Adds work-in-progress training script for forces
    
    * Updates MD-trainer to set grad_input targets
    
    * Adds MD notebook
    
    * Fixes `torch.no_grad` error when evaluating the forces model
    
    * Processes entire trajectory, not just first 100 samples
    
    * Adds edge distance calculation from positions to the autograd compute graph
    
    * Minor changes to support arbitrary devices
    
    * Updates distance from positions function to take arbitrary gaussian filter params
    
    * Adds position to batch.x if not present
    
    * Fixes normalization issue when computing forces via gradient of energies
    
    We were earlier normalizing forward and backward pass targets as
    (y - y_avg) / y_std. But this breaks analytical consistency between
    energies and forces (F = dE_dx). Instead, we normalize target energies
    same as before (E - E_avg) / E_std, and normalize forces as F / E_std.
    
    * added ase_calc template, resolved directory bug when "processed was not available"
    
    * ase calculator implementation and corresponding necessary changes
    
    * import order
    
    * Adding local 3D CNN model for CO_Cu_MD dataset
    
    * Fixes device issues to make it work on CPU
    
    * Transfer CNN3D embedding table to the correct device
    
    * Makes regress_forces compatible with earlier force models
    
    * Gasdb / trajectory dataloader returns atomic numbers, positions
    
    * Adjust input tensor dimensions if we're going to be appending positions later
    
    * suppress stdout for ase_calc simulations, dataloader for simulation
    
    * added .traj files to .gitignore, added ase md example
    
    * position naming bug
    
    * resolved training bug resulting form uneven weighting in loss function
    
    * added force_coefficient hyperparameter for weight balancing of energy/force contributions in loss function
    
    * support user defined loss function
    
    * added wandb to gitignore
    
    * Runtime and memory improvements to cnn3d_local model.
    
    * Fixes import order
    
    * mode flag added for predict, removed unneeded positions call, renamed ase calculator, resolved PR chages
    
    * pre-commit hooks
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: Larry Zitnick <zitnick@devfair0331.h2.fair>
    Former-commit-id: 4e29fa9a7a9ad762532694438a86edf6af49b1b5

[33mcommit 6d7ba525970967c8192a4eaa2d546b6e8117d3eb[m
Author: clz55 <52088675+clz55@users.noreply.github.com>
Date:   Mon Jun 1 23:16:11 2020 -0700

    Implements CO Cu dataloader, CNN3D model, trainer for force models (#11)
    
    * Adds dataloader for MD trajectories
    
    * Adds work-in-progress training script for forces
    
    * Updates MD-trainer to set grad_input targets
    
    * Adds MD notebook
    
    * Fixes `torch.no_grad` error when evaluating the forces model
    
    * Processes entire trajectory, not just first 100 samples
    
    * Adds edge distance calculation from positions to the autograd compute graph
    
    * Minor changes to support arbitrary devices
    
    * Updates distance from positions function to take arbitrary gaussian filter params
    
    * Adds position to batch.x if not present
    
    * Fixes normalization issue when computing forces via gradient of energies
    
    We were earlier normalizing forward and backward pass targets as
    (y - y_avg) / y_std. But this breaks analytical consistency between
    energies and forces (F = dE_dx). Instead, we normalize target energies
    same as before (E - E_avg) / E_std, and normalize forces as F / E_std.
    
    * Adding local 3D CNN model for CO_Cu_MD dataset
    
    * Fixes device issues to make it work on CPU
    
    * Transfer CNN3D embedding table to the correct device
    
    * Makes regress_forces compatible with earlier force models
    
    * Gasdb / trajectory dataloader returns atomic numbers, positions
    
    * Adjust input tensor dimensions if we're going to be appending positions later
    
    * Runtime and memory improvements to cnn3d_local model.
    
    * Fixes import order
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Co-authored-by: Larry Zitnick <zitnick@devfair0331.h2.fair>
    Former-commit-id: 2fd73b5b4a19e3be5a6aca5c064f5d3aceca990c

[33mcommit 0b472da89c0af53d1426c2174e3b72ffd2d60398[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Jun 1 11:26:36 2020 -0400

    bugfix: log avg train metrics over last 20 steps, not over all
    
    Former-commit-id: dcd23677bca81d21e038e882e2ef4853d854d714

[33mcommit 88594cb7a2884daa99df51aa28da6b798706d8f9[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri May 22 15:53:30 2020 -0400

    Install gpytorch via conda not pip
    
    
    Former-commit-id: 6f33499ff1729a15fe15525903173319f9634aa6

[33mcommit 00da1fd5bbf4043e4d976470eb296ae38ea8777d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue May 19 15:23:44 2020 -0400

    Don't evaluate on val / test splits if they are zero size
    
    
    Former-commit-id: 6bc406d4b89b435b588873d76f9bf6793284fc7c

[33mcommit edeecddfd3a9c0cbd8bb47a502a68d19caf56d72[m
Author: Muhammed Shuaibi <45150244+mshuaibii@users.noreply.github.com>
Date:   Fri May 15 17:25:15 2020 -0500

    included ray in dependencies
    
    Tune install is required as part of `trainers/__init__.py`
    
    Former-commit-id: 35db09b110d48e5b03ff37662c22a32358e116db

[33mcommit 61e1a44cdc0179c4cc9675d1136b02373bb9ac2a[m
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Wed May 13 15:44:08 2020 -0400

    Enable parsing of GASdb energy data from multiple database locations
    
    
    Former-commit-id: fd61129dcc4aad8a81d5cb5048e6acf35f52c894

[33mcommit 469532923f54e750c0ceea5cab6c7add0adf32a3[m
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Wed May 13 14:39:26 2020 -0400

    Implement Convolution-Fed Gaussian Process (CFGP) (#7)
    
    * Initial framework for GP training via GPyTorch
    
    * Remove inter-trainer dependency from GPyTorch trainer
    
    * Feed training data to GPyTorchTrainer during training instead of initialization
    
    * Initial framework for CFGP
    
    * Debug CFGP
    
    * Changes relative to absolute imports
    
    * Adds gpytorch install note in readme + minor reorganization
    
    * Reorganize CFGP code for easier debugging
    
    * Fix CFGP bug with shuffling
    
    We got the convolutions and the targets separately. But everytime we
    called the train loader, the data was reshuffled. So we were trying to
    get the GP to train on data where the X's and Y's were not actually
    mapping to each other correctly. This commit fixes that.
    
    * Adds a CFGP runner example from Kevin's jupyter code snippets
    
    * Align formatting for gpytorch trainer predictions
    
    * Add Dockerfile to build image
    
    * Specify version for skimage
    
    * Install RayTune via pip instead of conda
    
    * Expose /home/.config to mounting to enable wandb
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>
    Former-commit-id: 59a93391cabf3be734fd1bebd31b186b0ecbdf63

[33mcommit c9c915e461bb89ad547f1a5b41c1b68bbc088f65[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri May 8 16:40:47 2020 -0400

    Implements Schnet-style model with CGCNN to jointly predict energies and forces (#8)
    
    * Adds dataloader for MD trajectories
    
    * Adds work-in-progress training script for forces
    
    * Updates MD-trainer to set grad_input targets
    
    * Adds MD notebook
    
    * Fixes `torch.no_grad` error when evaluating the forces model
    
    * Processes entire trajectory, not just first 100 samples
    
    * Adds edge distance calculation from positions to the autograd compute graph
    
    * Minor changes to support arbitrary devices
    
    * Updates distance from positions function to take arbitrary gaussian filter params
    
    * Adds position to batch.x if not present
    
    * Fixes normalization issue when computing forces via gradient of energies
    
    We were earlier normalizing forward and backward pass targets as
    (y - y_avg) / y_std. But this breaks analytical consistency between
    energies and forces (F = dE_dx). Instead, we normalize target energies
    same as before (E - E_avg) / E_std, and normalize forces as F / E_std.
    
    Former-commit-id: f611d8680a8862c18a06ea424082b7c479cfb924

[33mcommit fc33488bd0c8fe9c66f51a5d34601b32d8065bf3[m
Author: Brandon Wood <b.wood@berkeley.edu>
Date:   Fri May 8 13:36:42 2020 -0700

    Hpo Ray Tune (#9)
    
    * hpo branch first commit
    
    * tune update with BaseDataset used
    
    * path added to run hpo and small whitespace issue fixed in train_hpo
    
    * Refactoring tune hyperparam sweeps to build on BaseTrainer
    
    * updated hpo to move parameters to config file
    
    * pre-commit changes
    
    * removed abs paths from YAML files and made a few other small changes per PR
    
    Co-authored-by: Abhishek Das <das.abhshk@gmail.com>

[33mcommit 0cadfed0f7cde988bce0b8a8b8b7fdd7717dca54[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue May 5 19:51:48 2020 -0400

    Fixes bug in computing normalization params over the training set

[33mcommit bc9c1e6741fc3d507dd9638450de54c3f40ce310[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Sat May 2 14:26:09 2020 -0400

    reproducing doggs results

[33mcommit 79b369bb507e5ccf02034741d831180945e0be4b[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Fri May 1 17:43:41 2020 -0400

    shuffle dataset for DOGGS

[33mcommit a96909063d0cad5f2a1c54ab48ad81daa9539fca[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Thu Apr 30 12:45:21 2020 -0400

    benchmark for surface dataset

[33mcommit 5a91ca857edf9b1ec87d11935cf192e51654dbc3[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Thu Apr 30 12:44:37 2020 -0400

    updated DOGSS conv layer and optimizer

[33mcommit 273c9b9719870992ccc8812bc730fe48b5b284d6[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Thu Apr 30 12:43:19 2020 -0400

    remove preprocessign folder

[33mcommit 234c2015e55242091a2529db27f60f672f54b023[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Wed Apr 29 23:50:54 2020 -0400

    test run

[33mcommit 7425be8f3ab0d46708242dc37cfebc5b0198d0ab[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Wed Apr 29 23:50:17 2020 -0400

    cleaned DOGSS code and adeded few features

[33mcommit 3bf818838d54e8798e05930c44a73981f21cb060[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Apr 27 16:43:12 2020 -0400

    Removes stray untitled.txt

[33mcommit 9641cf5ee957196f6bfc69887bd7af8a4382cdd2[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Apr 21 19:47:56 2020 -0400

    Refactors DOGSS code: runs on CPU/GPU, disables normalization, lints

[33mcommit 8f994829ea99b10ed13c187551eec30565fc3bda[m
Merge: 4c7a1ad 06ea5a3
Author: junwoony <junwoongyoon@gmail.com>
Date:   Tue Apr 21 01:43:18 2020 -0400

    Merge branch 'master' of https://github.com/Open-Catalyst-Project/baselines

[33mcommit 4c7a1ad3b33faadb9140cd9f5ef5c8cb81ec80bc[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Tue Apr 21 01:40:49 2020 -0400

    Adding DOGSS: updated model, trainer, metrics, optim, etc.

[33mcommit 06ea5a305b380bc2fa177d57a06f340eaa3298a6[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Apr 20 23:40:55 2020 -0400

    Updates installation instructions

[33mcommit 88bcc9fa4f0301808b8921d621b023de24141dad[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Apr 20 18:47:09 2020 -0400

    Removes relative imports from datasets.gasdb

[33mcommit 614dc122132e25bff9b46efb4f8634a226b26349[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Apr 20 18:02:04 2020 -0400

    Fixes bug and explicitly checks for no. of neighbors in gasdb `AtomicFeatureGenerator`

[33mcommit 873fccabaa7de492a460bd04e189568b1f458f97[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Sat Apr 18 03:55:24 2020 -0400

    uploading preprocessing model (DOGSS) in the old version of code (need to be reformatted/rewritten)

[33mcommit 2dd4f7e575e930d0e786ac3e37da90d1fc307b12[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Sat Apr 18 03:49:29 2020 -0400

    uploading preprocessing model (Need to be rerwote and cleaned up)

[33mcommit ee9f27dc7b9cbe68d377af8051494dc6d174abcc[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Apr 16 03:15:45 2020 -0400

    Restores normalization while making predictions

[33mcommit f789bc48c4ea9e69bfab3b3dd6c8e53c954b4e19[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Apr 16 03:02:18 2020 -0400

    Updates readme to have instructions for latest workflow

[33mcommit edbd11c5d3355e13859725f780f55d3403b6d0a7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Apr 16 02:44:36 2020 -0400

    Fixes `ocpmodels` import from example training script

[33mcommit e4ed492986633271d13ad142659c753b2f68aeb9[m
Merge: e5ae56c 8069464
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Apr 16 02:35:55 2020 -0400

    Merge branch 'ulissigroup-sktrainer'

[33mcommit 8069464933c01746692fa79dcdb2411cb3b717bb[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Apr 16 02:30:17 2020 -0400

    Renames and minor fixes to `SimpleTrainer`; adds an example training script

[33mcommit f1b7cdc0aab21d60b135e641986f36d44d05b2eb[m
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Wed Apr 15 17:27:35 2020 -0400

    Implement predict method for SKTrainer

[33mcommit 1e38f87173cf565d4806352be942c11168f1edf7[m
Merge: 08faeee e5ae56c
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Wed Apr 15 16:20:49 2020 -0400

    Merge branch 'master' into sktrainer

[33mcommit e5ae56cdf342f1d7087ddd936eb9d34cfa3eceab[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Apr 15 15:08:46 2020 -0400

    Fixes silently failing ValueError when zipping atom neighbor indices and distances (#4)

[33mcommit 08faeee00980c03faca0181324c5beea8b33573b[m
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Tue Apr 14 15:50:14 2020 -0400

    Make log directory pointing more consistent

[33mcommit 4f1b7f0fc8b6b79d79739feca452237d28398231[m
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Tue Apr 14 15:31:49 2020 -0400

    Simplified SKTrainer

[33mcommit 509b2646a63f22dd9cba5f5ca819073e2e2312e7[m
Author: Kevin Tran <ktrain9891@gmail.com>
Date:   Tue Apr 14 13:00:59 2020 -0400

    Initial framework for SKLearn-like trainer

[33mcommit 834af16a150cee7191567574f6b3f75e7c9f7548[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Apr 8 19:27:31 2020 -0400

    Renames library from `baselines` to `ocpmodels`

[33mcommit 1414c7e4448ad1236c34033ed9b63e36bd4b2c0d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Apr 8 19:08:32 2020 -0400

    Implements a modified version of CGCNN from Gu et al., 2020 (#2)
    
    * Implements the modified CGCNN from Gu et al: https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c00634
    
    Changes to the architecture from CGCNN:
    - Tanh instead of SoftPlus in the conv layers
    - Attention layer to compute weights over atoms before pooling
    
    * Fixes naming conflict with recent version of pytorch-geometric

[33mcommit 4a4e6a125d3055b67af0a03573ae0cf291b88ddb[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Apr 8 17:17:35 2020 -0400

    Integrates GASdb dataset (#1)
    
    commit b62a1868c6dcd1e0c652190f082d45d9284b6d08
    Author: Abhishek Das <das.abhshk@gmail.com>
    Date:   Wed Apr 8 17:11:04 2020 -0400
    
        Runs black + isort
    
    commit ee433fbcb4abf119660941cac2e2e2c4ac171a10
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Wed Apr 8 15:42:24 2020 -0400
    
        Address comments in PR #26
    
    commit 486c837f11b13f0182537375b3b4e5c730c73700
    Merge: b55579e 9ebcd26
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Wed Apr 8 13:46:44 2020 -0400
    
        Merge branch 'gasdb' of github.com:Open-Catalyst-Project/baselines into gasdb
    
    commit 5f25e8d734e97db85d9b111e0d3ad46d2c8e9e09
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Wed Apr 8 13:40:34 2020 -0400
    
        Fix length bug with GASdb dataset
    
    commit 90c3d1484f457d55a7da508e011dc2b2f95e481d
    Author: Abhishek Das <das.abhshk@gmail.com>
    Date:   Mon Apr 6 17:23:54 2020 -0400
    
        Don't override BaseDataset's `__len__` method; messes up indexing otherwise
    
    commit f0496744bb78e63283553eb4a930ac7b0b4ff059
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Mon Apr 6 15:33:05 2020 -0400
    
        Convert GASdb dataset to InMemoryDataset
    
    commit a1cbdfc9d4bd0be266efd8bd485250941df9575e
    Merge: 698874d ccaac44
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Fri Apr 3 14:18:40 2020 -0400
    
        Merge branch 'master' into gasdb
    
    commit bd816f4f96ea4bed23c7dbcdfa9a197bdf00e3cb
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Tue Mar 31 16:56:55 2020 -0400
    
        Debugging GASdb pipeline
    
    commit 6f190897f9e8092eb19e2cd44df47a0f878150d5
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Fri Mar 27 16:27:47 2020 -0400
    
        Initial framework for CFGP fit on GASdb
    
    commit 1289de7b6c438434ec357e840e3ec9bd743c211e
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Fri Mar 27 11:20:43 2020 -0400
    
        Give the convolution step in CGCNN its own method
    
        So that we can call on it later to train a convolution-fed Gaussian
        process regressor.
    
    commit 3fa810767c68fbd81b9e4696c85966f39113f21c
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Fri Mar 27 11:19:58 2020 -0400
    
        Start ignoring VIM swap files
    
    commit fb9fc3911fb088b1699d8ac5a724ae40d13de7db
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Fri Mar 27 11:19:18 2020 -0400
    
        Rename active discovery trainer to more generic ase trainer
    
    commit 6ab9df9fb3e70d709398169fed6ead38721754d3
    Merge: 5ce0e52 69ff54b
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Tue Mar 24 16:37:01 2020 -0400
    
        Merge branch 'active-learning' of github.com:Open-Catalyst-Project/baselines into active-learning
    
    commit 86f20460166688e1d32bc7191372edc24cf57568
    Merge: 67cd9b3 12fc2c4
    Author: Abhishek Das <das.abhshk@gmail.com>
    Date:   Tue Mar 24 16:35:54 2020 -0400
    
        Merge branch 'master' into active-learning
    
    commit 99de5f2e8272c1ed489f24c625a0e824e3df7b7d
    Author: Kevin Tran <ktrain9891@gmail.com>
    Date:   Tue Mar 24 14:28:32 2020 -0400
    
        Get rid of circular dependency
    
    commit 5fb03b1c27f74a06fd2769e159d1ce5421423d24
    Author: Abhishek Das <das.abhshk@gmail.com>
    Date:   Tue Mar 3 00:38:09 2020 -0500
    
        Initial boilerplate code for active learning setup

[33mcommit ac27cfd0d11c783fa2d7867efbf886d9df98dad0[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Apr 1 19:24:54 2020 -0400

    Fixes naming conflict with recent version of pytorch-geometric

[33mcommit e4fe27cbc5d7e20b482dab510e356efc89fd327b[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Mar 16 00:44:47 2020 -0400

    Example jupyter notebook for training CGCNN on XieGrossmanMatProj

[33mcommit 65801440ad427ea9063496fa785cf6f76ca91e29[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Feb 19 02:42:25 2020 -0500

    Adds dropout to attention computation in AttentionConv

[33mcommit 4cf7bb56cde921d9a561358eb4128ca250a4c299[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Feb 18 20:45:09 2020 -0500

    5 attention heads works well for Transformer on MatProj

[33mcommit f8b9b7173697c225f3ddc7ade7c3456f17c77a61[m
Author: junwoony <junwoongyoon@gmail.com>
Date:   Mon Feb 17 20:18:40 2020 -0500

    updated geometric.py and script to create graph from docs

[33mcommit 3db97baa5a91359c0b32dde11b6802b18f3d1dfc[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Feb 17 17:39:09 2020 -0500

    Adds script to convert UlissigroupCO data to PyG format

[33mcommit 90051ac3d661a47c58579980715cc64ddb304659[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Feb 17 14:21:45 2020 -0500

    Better initial lr for transformer

[33mcommit 76b0ce5ebe4dbac88b951eb8394aeea4f3f63727[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Feb 17 00:38:49 2020 -0500

    Implements graph transformer network from https://arxiv.org/abs/1905.12712.
    
    Also related to https://openreview.net/forum?id=HJei-2RcK7 and
    https://arxiv.org/abs/1710.10903.

[33mcommit 8587dcd8573b91b0bd6650c9cbf74f0ca8410c5c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Feb 16 01:15:24 2020 -0500

    Plots data target distributions before training run via `vis` param

[33mcommit 31c5a25011d586a1a6052ae99d91c73077e3a418[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Feb 13 19:15:53 2020 -0500

    Fixes typo

[33mcommit 31389735923530d57adb87df23e220ef4fcb5405[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Feb 13 18:57:26 2020 -0500

    cgcnn --> baselines

[33mcommit c4322ba76031dd8df7f589eeffce04611c150854[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Feb 13 04:35:08 2020 -0500

    Cleaner data splits; fixed test split

[33mcommit 882a65efc7b3f93c8710f2913c7013bda1caa725[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Feb 13 02:52:10 2020 -0500

    Gets back checkpointing

[33mcommit 527e59da0cfc396080ca0b160bff645b61b03bfa[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Feb 12 22:51:18 2020 -0500

    Uses labels as keys when logging

[33mcommit 1a5e7902501531ba4b75971ed625786c55760820[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Feb 11 22:56:05 2020 -0500

    Abstracts out base dataset to a separate class

[33mcommit 420fe341bdece163529e3bd38cc89236944dec6e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Feb 11 22:22:27 2020 -0500

    bugfix: corrects epoch that gets printed to stdout

[33mcommit 4e10b39116788e690f9c83d19fcec622d41678a7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Feb 11 22:12:35 2020 -0500

    Abstracts out base model into separate class

[33mcommit 550fcb2eca4e7c009f9502108cd1fe68dbb18211[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Feb 11 21:54:59 2020 -0500

    bugfix: fixes conv layer dimension for QM9

[33mcommit 1e12b1f4da06b332680585d49315529bbfdc91e5[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Feb 11 17:13:44 2020 -0500

    Logging updates: eval on test, log run identifiers

[33mcommit 311eb4281da041ca44772564b0996b59b415ccef[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Feb 10 21:40:52 2020 -0500

    Global registry; supports generic logging

[33mcommit a165e8e7d66d2bfebf2a1534b94210f82c481794[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Feb 10 17:50:18 2020 -0500

    Abstracts logger into separate class; integrates with wandb

[33mcommit aacab44ae23e8d50c87bc214dac41d9b609b08ec[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Feb 9 22:18:10 2020 -0500

    Major refactor: abstracted out trainer to separate class

[33mcommit 684930d4b5dbec52f154ee78415ef1f285850700[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jan 29 21:14:19 2020 -0500

    Computing derivatives to predict atomic forces on ISO17

[33mcommit 5739980112bfbf22ac9bfdb2553b979061d64629[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Jan 29 17:18:01 2020 -0500

    Support for testing on `test_within` and `test_other` for ISO17
    
    test_within: 20% unseen steps of seen MD trajectories
    test_other: remaining 20% unseen MD trajectories

[33mcommit e120d9ebcafcadeb0bf3e645164eec42091db950[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jan 28 23:29:42 2020 -0500

    Working version of ISO17 energy prediction model

[33mcommit 2e9a12b21b543322268ab0709e28470f1572af5d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Jan 21 17:02:37 2020 -0500

    Adds tqdm to requirements

[33mcommit 8e24088399ba8b3b0adcf361c4c945f3a6528136[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 29 18:01:13 2019 -0400

    Adds ISO17 download link; ase for parsing it

[33mcommit c72e212311afe81815bdf2ddf4fd6d78700ef8a4[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 29 18:00:30 2019 -0400

    Adds pre-commit to requirements

[33mcommit 5d59e9c9f7c9006c3c8340207bc4a339d8cf2776[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 26 01:06:01 2019 -0400

    Slimmer set of requirements

[33mcommit 30c28699449ca2ff8aa4598f31ef20b4c1a81ff0[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Oct 16 02:16:13 2019 -0400

    bugfix in edge dimension

[33mcommit 36c5fb3ce9791a6b9866b6aea0c7b3a14564f0a0[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 15 22:45:34 2019 -0400

    Adds distance to QM9 edge attributes

[33mcommit 150874d3447fe9aba7eef8d07f25e7101f19f22f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 15 21:28:54 2019 -0400

    Compute the mae metric for qm9

[33mcommit be7efdef691311406fd449fa338b5d4fdb2fb5b7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 16:27:35 2019 -0400

    Adds demjson to deps

[33mcommit 3e6c640153c9189870abc3dff0e15a188d556a83[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 16:25:07 2019 -0400

    Option to pass `label_index` as false for QM9

[33mcommit 566c510356ee6dd8faf46c78818c78e19f34dbd8[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 16:24:33 2019 -0400

    Fixes minor memory leak

[33mcommit f74275957bc86ad3b0d06d97cf690ec21f423d54[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 15:54:34 2019 -0400

    Support for overriding config-yml params

[33mcommit 306b7e3659f2c2c13579fb043a93602edf58b113[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 02:32:55 2019 -0400

    Removes unused code

[33mcommit 0d70a4d0acd54e44cdd27e43d5e5bda48c25c07e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 02:26:35 2019 -0400

    Unzip in data/data

[33mcommit 3080d26640966bc1e64f3097ee3402ddfc76a6c3[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 02:12:40 2019 -0400

    Support for QM9

[33mcommit 592cb03766f9ed4a7f74b5ff5d6d12d77e0bc572[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 01:23:52 2019 -0400

    torch-geometric dependencies

[33mcommit bb94cbbbf55112e60a4e1e189265d8742d25aa9e[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 01:21:22 2019 -0400

    Support for Xie and Grossman's Matproj split

[33mcommit 650c482f167e120eebe86576a06893c77cb4448f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 14 00:00:03 2019 -0400

    Ported over CGCNN to be pytorch-geometric compatible

[33mcommit 584c9e2b29c60fbe4aa95e5487bc9129e203a123[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 12 21:50:33 2019 -0400

    Warmup learning rate in first few epochs

[33mcommit 86e4e8352014a1ed661c2d4c102afb689ffbcd1a[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Fri Oct 11 00:28:28 2019 -0400

    Updates dataset instructions in readme

[33mcommit d25ba5a971faf8e2cdd3a608f3d78fa4d6d645ab[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 10 23:23:33 2019 -0400

    Preprocessed material project dataset

[33mcommit abbc5cd4eb546b3b4e2a4f0895d013d8fd28a2fd[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 10 23:19:13 2019 -0400

    pytorch 0.4.1 --> 1.2.0

[33mcommit d775120d6edf61756f0d1148beeec83bbe6347fd[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 10 16:48:39 2019 -0400

    Dataloader for Xie and Grossman's Material Project dataset

[33mcommit 771ed7c27fc3fb268b7b22302e3554784d54876d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Oct 10 16:23:31 2019 -0400

    Working version of qm9

[33mcommit f09862bd5cc6de766ca4f80846fe1f1f7ca82e50[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 8 19:09:40 2019 -0400

    Documenting a few dims in the model code

[33mcommit 4f0815308577f3393ac13fb0e24522441377508f[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 8 17:37:43 2019 -0400

    Dumping the visualization jupyter notebook

[33mcommit 407d344e5197a9da08d03c0bb322932305aa9826[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Tue Oct 8 01:30:38 2019 -0400

    Better logging, smaller batch size, minor code refactor

[33mcommit 720a5e04437e68d0c24c4e4c3cb90992485693ac[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Mon Oct 7 23:23:12 2019 -0400

    Pass seed as command-line param

[33mcommit ff3f20085d220320b4b067bfdd266c07b82a67bb[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 6 16:20:21 2019 -0400

    typo

[33mcommit 54a00abf6d49c5ef2bed5afe8b1a9b6ba4cd3bb2[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 6 04:08:37 2019 -0400

    Adds download link to QM9

[33mcommit b452f53372db8e32ade563d2d1085518d72d0157[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 6 03:49:06 2019 -0400

    torch version note

[33mcommit 57f630505f8419695ca6eeff8e7e74e3ee0362f8[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 6 03:37:35 2019 -0400

    Remove legacy data_module

[33mcommit 2c51ade9185c67989e9cfbaea29662481f8479ab[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 6 03:36:09 2019 -0400

    Readme update

[33mcommit 845730a281fd67a018a6e7855f33464b22c2d632[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 6 03:17:10 2019 -0400

    Deleting legacy folders

[33mcommit 56236c5f07ffb20be744a84328c71d5e2c0bbccf[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sun Oct 6 00:46:53 2019 -0400

    Major refactor to (eventually) support multiple datasets / tasks

[33mcommit 102d33e2c2dd74acadd978f5c662bf6187d95212[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 21:52:20 2019 -0400

    Gradient clipping seems to help a bit

[33mcommit 0e54475b22f93ebcb099f28f01212463b3f4d6c3[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 17:24:30 2019 -0400

    Log learning rate

[33mcommit f3f9df7c58bac274338671fbb6e78188ecdb647d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 17:08:41 2019 -0400

    isort

[33mcommit 7f0aa217689883c3a8d9eacadf72914660680c21[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 16:07:58 2019 -0400

    Better defaults

[33mcommit a1b955db498526fb334cb0df16b41814d3b1e7cf[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 02:44:36 2019 -0400

    Better defaults hyperparams

[33mcommit 5d4d1bce0d33291de6185a96f56a244dde4771b8[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 01:50:06 2019 -0400

    Checkpointing with identifier, saving configs, minor bugfixes

[33mcommit 9488b22aaa35cf3ed4d834aa6c3ec7d2c5558eab[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 00:23:09 2019 -0400

    Path bug fix

[33mcommit 4fa380f4984a7f82ba63f625350b504876c7ceaf[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 00:21:41 2019 -0400

    black, flake8

[33mcommit c839e26099e7b3a783470bd3ded68804fea41bed[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Sat Oct 5 00:19:14 2019 -0400

    Now getting val MAE of ~0.145 for the full case

[33mcommit a979e2e3c1c63a9ab0097ad609fa22463b22c921[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Oct 2 01:50:16 2019 -0400

    Add log of experiments; update lr, loss

[33mcommit 91df4b544b97d81a0985a8af0fb8b4aadad96f2d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Wed Oct 2 01:49:45 2019 -0400

    Update requirements; adds skorch and related deps

[33mcommit 6d3115bf27746c081c45c3f2c1de040f2f863ec7[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 26 17:42:31 2019 -0400

    Adam, lr1e-3

[33mcommit 04867fc38b8a55f8cc7905ba0eca9f87377a05eb[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 26 15:32:54 2019 -0400

    Updated dataset

[33mcommit f9d95f20c56557aadb6901c98058a0ce44dff5c6[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 26 14:10:27 2019 -0400

    Minor path-related bug fixes

[33mcommit 6cf7f8949d569b2469932e1d63f9330ee797ecb1[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 26 02:51:00 2019 -0400

    Better logging, checkpointing, tensorboard support

[33mcommit a2e9815189ef99a4f10867cdab404a44983cda8c[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 26 01:48:04 2019 -0400

    Instructions for setting up the dataset

[33mcommit 8149cf8db7327e03f38bbe6b567e17b08bf4235d[m
Author: Abhishek Das <das.abhshk@gmail.com>
Date:   Thu Sep 26 01:17:32 2019 -0400

    Updates setup + readme

[33mcommit b7655edc4d6e82007ca05f4290e40df939a658dc[m
Author: nianhant <nianhant@andrew.cmu.edu>
Date:   Thu Nov 8 15:09:07 2018 -0500

    Optimize CGCNN via SIGOPT

[33mcommit e532d59e099f629e089744a84c428b27c1ed9536[m
Author: Kaylee Tian <35741285+nianhant@users.noreply.github.com>
Date:   Mon Oct 22 16:16:42 2018 -0400

    Add files via upload
    
    Added VoronoiConnectivity to all_nbrs

[33mcommit d09d4368fc2ef4b1f118eaf8b7832059122aac7e[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Sun Oct 7 22:19:00 2018 -0400

    Fixed the erroneous efermi pre-trained model

[33mcommit fadbde836b9d97823d17519302951c6bd5240962[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Fri Jul 27 14:30:18 2018 -0400

    Print warnings instead of errors if not enough neigbhors

[33mcommit c16cd14c4a9790cab8b659db750ceaf3dd4d0cd0[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Mon May 28 14:08:40 2018 -0400

    Updated the pre-trained table

[33mcommit 9c0f8998655353f113b29fcdc1fe398ed61071bd[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Sun May 27 01:52:59 2018 -0400

    Updated the pre trained model for band gap

[33mcommit 52a61debb4b46f51cc78f021e5c76bdf18c439db[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Fri Apr 27 14:43:52 2018 -0400

    Noted the breaking changes caused by pytorch v0.4.0

[33mcommit d575906a06ecdced1c37af2281d45bec657cd001[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Fri Apr 6 14:12:46 2018 -0400

    Updated citations

[33mcommit afedd86fe5c0c3cc4611b00cbc1393bbc9f8a532[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Thu Mar 15 15:45:05 2018 -0400

    Fixed a typo

[33mcommit 704d0ad390f7cf7d2967e4a5ba31bb1a8e2df453[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Thu Mar 15 15:20:12 2018 -0400

    docs for CIFData

[33mcommit 2b194299ef4725a6a8359770f06f9ee04246c678[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Wed Mar 14 20:36:45 2018 -0400

    Fixed a line break

[33mcommit c3d46d4130b663a3d6500a3444a5d98f5eaa565f[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Wed Mar 14 20:20:15 2018 -0400

    Fixed header links

[33mcommit 1d351d76184556bfa5f6adce910175ed31a040d4[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Wed Mar 14 20:12:40 2018 -0400

    Added README for pre-trained models

[33mcommit 5e582681195f16896dce5036da36ecdd3c2f3fe5[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Wed Mar 14 18:36:31 2018 -0400

    Added table of contents and cross references

[33mcommit 519b7460efb51f4afdf1c539c87bdb39c23fbdc4[m
Author: Tian Xie <tian.tim.xie@gmail.com>
Date:   Wed Mar 14 16:41:56 2018 -0400

    Initial commit
