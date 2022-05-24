#!/usr/bin/env python3

setup(
    cmdclass={
    name='ZeusGAN',
    version='1.0',
    description='GAN for Zeus',
    author='Craig MacLachlan',
    author_email='cs.maclachlan@gmail.com',
    #packages=['articlass'],
    python_requires='>=3.7',
    install_requires=['tensorflow==2.6.4',
                      'matplotlib',
                      'numpy',
                      'scipy',
                      'imageio'],
    )