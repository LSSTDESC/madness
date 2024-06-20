"""Test Deblending."""

import numpy as np

from madness_deblender.deblender import Deblender, compute_residual


def test_deblending():
    """Test deblending."""
    deb = Deblender(
        stamp_shape=5,
        latent_dim=4,
        filters_encoder=[1, 1, 1, 1],
        filters_decoder=[1, 1, 1],
        kernels_encoder=[1, 1, 1, 1],
        kernels_decoder=[1, 1, 1],
        dense_layer_units=1,
        num_nf_layers=1,
        load_weights=False,
    )

    data = np.random.rand(2, 15, 15, 6)

    detected_pos = [[[9, 10], [11, 11]], [[10, 10], [0, 0]]]

    deb(
        blended_fields=data.copy(),
        detected_positions=detected_pos,
        num_components=[2, 1],
        use_log_prob=True,
        linear_norm_coeff=1,
        max_iter=2,
        use_debvader=True,
        map_solution=True,
        channel_last=True,
    )

    deb(
        np.moveaxis(data.copy(), -1, -3),
        detected_pos,
        num_components=[2, 1],
        use_log_prob=True,
        linear_norm_coeff=1,
        max_iter=2,
        use_debvader=True,
        map_solution=True,
        channel_last=False,
    )

    index_pos_to_sub = deb.get_index_pos_to_sub()
    residual1 = compute_residual(
        blended_field=data[0],
        reconstructions=np.moveaxis(deb.get_components()[0], -3, -1),
        use_scatter_and_sub=True,
        index_pos_to_sub=index_pos_to_sub[0],
    ).numpy()

    padding_infos = deb.get_padding_infos()
    residual2 = compute_residual(
        blended_field=data[0],
        reconstructions=np.moveaxis(deb.get_components()[0], -3, -1),
        use_scatter_and_sub=False,
        padding_infos=padding_infos[0],
    ).numpy()

    np.testing.assert_array_equal(residual1, residual2)


def test_scatter_and_sub():
    """Test scatter and sub."""
    deb = Deblender(
        stamp_shape=5,
        latent_dim=4,
        filters_encoder=[1, 1, 1, 1],
        filters_decoder=[1, 1, 1],
        kernels_encoder=[1, 1, 1, 1],
        kernels_decoder=[1, 1, 1],
        dense_layer_units=1,
        num_nf_layers=1,
        load_weights=False,
    )

    data = np.random.rand(1, 15, 15, 6)

    detected_pos = [[10, 10]]

    deb(
        data.copy(),
        [detected_pos],
        num_components=[len(detected_pos)],  # redundant parameter
        use_log_prob=True,
        linear_norm_coeff=1,
        max_iter=2,
        use_debvader=True,
        map_solution=True,
        channel_last=True,
    )

    index_pos_to_sub = deb.get_index_pos_to_sub()
    residual1 = compute_residual(
        blended_field=data[0],
        reconstructions=deb.get_components()[0],
        use_scatter_and_sub=True,
        index_pos_to_sub=index_pos_to_sub[0],
    ).numpy()
    padding_infos = deb.get_padding_infos()
    residual2 = compute_residual(
        blended_field=data[0],
        reconstructions=deb.get_components()[0],
        use_scatter_and_sub=False,
        padding_infos=padding_infos[0],
    ).numpy()

    np.testing.assert_array_equal(residual1, residual2)
