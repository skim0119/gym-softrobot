"""Rendering tests require optional system-level renderer dependencies."""

# FIXME: Testing rendering is disabled for now. It needs to install POVRAY on CI
#@pytest.mark.parametrize("spec", spec_list)
#def test_env_render_result_np_array_for_rgb_mode(spec):
#    env = spec.make()
#    env.reset()
#    output = env.render(mode='rgb_array')
#    assert isinstance(output, np.ndarray)
#    assert output.shape[2] == 3
#    env.close()
