# MADNESS Deblender
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
![PyPI - Version](https://img.shields.io/pypi/v/madness-deblender)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![pytest](https://github.com/b-biswas/madness/actions/workflows/pytest.yml/badge.svg)](https://github.com/b-biswas/madness/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/b-biswas/madness/graph/badge.svg?token=2YNVQ7C4CU)](https://codecov.io/gh/b-biswas/madness)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![All Contributors](https://img.shields.io/badge/all_contributors-6-blue.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

Maximum A posteriori with Deep NEural networks for Source Separation [arXiv:2408.15236](https://arxiv.org/abs/2408.15236v1)

This repository contains the code for the MADNESS project under development within the LSST Dark Energy Science Collaboration (LSST DESC).
MADNESS obtains the MAP solution to deblend galaxies in a blended scene by performing gradient descent in a VAE latent space.

## Installation
For testing the deblender, the package can be installed directly from GitHub
```bash
 pip install madness-deblender
```
For contributing (further instructions to be added soon):
```bash
git clone https://github.com/LSSTDESC/madness/
cd madness
pip install -e .[dev]
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/b-biswas"><img src="https://avatars.githubusercontent.com/u/44917825?v=4?s=100" width="100px;" alt="Biswajit Biswas"/><br /><sub><b>Biswajit Biswas</b></sub></a><br /><a href="https://github.com/b-biswas/MADNESS/commits?author=b-biswas" title="Code">💻</a> <a href="#ideas-b-biswas" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/b-biswas/MADNESS/commits?author=b-biswas" title="Tests">⚠️</a> <a href="#maintenance-b-biswas" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://junpenglao.xyz/"><img src="https://avatars.githubusercontent.com/u/12952641?v=4?s=100" width="100px;" alt="Junpeng Lao"/><br /><sub><b>Junpeng Lao</b></sub></a><br /><a href="https://github.com/b-biswas/MADNESS/commits?author=junpenglao" title="Code">💻</a> <a href="https://github.com/b-biswas/MADNESS/pulls?q=is%3Apr+reviewed-by%3Ajunpenglao" title="Reviewed Pull Requests">👀</a> <a href="#ideas-junpenglao" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://aboucaud.github.io"><img src="https://avatars.githubusercontent.com/u/3065310?v=4?s=100" width="100px;" alt="Alexandre Boucaud"/><br /><sub><b>Alexandre Boucaud</b></sub></a><br /><a href="https://github.com/b-biswas/MADNESS/pulls?q=is%3Apr+reviewed-by%3Aaboucaud" title="Reviewed Pull Requests">👀</a> <a href="#ideas-aboucaud" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aubourg"><img src="https://avatars.githubusercontent.com/u/4321664?v=4?s=100" width="100px;" alt="Eric Aubourg"/><br /><sub><b>Eric Aubourg</b></sub></a><br /><a href="#ideas-aubourg" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/roucelle"><img src="https://avatars.githubusercontent.com/u/17788009?v=4?s=100" width="100px;" alt="Cécile Roucelle"/><br /><sub><b>Cécile Roucelle</b></sub></a><br /><a href="#ideas-roucelle" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aguinot"><img src="https://avatars.githubusercontent.com/u/39480528?v=4?s=100" width="100px;" alt="Axel Guinot"/><br /><sub><b>Axel Guinot</b></sub></a><br /><a href="#ideas-aguinot" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License
MIT
