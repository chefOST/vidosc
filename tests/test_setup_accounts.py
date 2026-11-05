import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import setup_accounts


@pytest.fixture
def accounts(monkeypatch):
    secrets = SimpleNamespace(list=Mock(return_value=[SimpleNamespace(name='chefost-r2')]), create=Mock())
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(Secret=SimpleNamespace(objects=secrets)))
    config = SimpleNamespace(get=Mock(return_value='dev'))
    monkeypatch.setitem(sys.modules, 'modal.config', SimpleNamespace(config=config))
    monkeypatch.setattr(setup_accounts, 'modal_workspace', lambda: 'personal')
    api = SimpleNamespace(viewer=object(), project=Mock(return_value=object()))
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Api=Mock(return_value=api)))
    frozen = SimpleNamespace(access_key='private-id', secret_key='private-key', token=None)
    session = Mock(return_value=SimpleNamespace(get_credentials=lambda: SimpleNamespace(get_frozen_credentials=lambda: frozen)))
    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(Session=session))
    return secrets, config, session


def test_read_only_check_uses_active_modal_environment(accounts):
    secrets, config, session = accounts
    setup_accounts.check_accounts()
    config.get.assert_called_once_with('environment')
    secrets.list.assert_called_once_with(environment_name='dev')
    secrets.create.assert_not_called()
    session.assert_not_called()


def test_wrong_workspace_never_reads_or_transmits_r2_credentials(accounts):
    secrets, _, session = accounts
    with pytest.raises(RuntimeError, match='expected another'):
        setup_accounts.check_accounts(True, expected_workspace='another')
    secrets.list.assert_not_called()
    secrets.create.assert_not_called()
    session.assert_not_called()


@pytest.mark.parametrize('create,workspace', [(False, None), (True, None)])
def test_missing_secret_requires_explicit_creation_and_workspace(accounts, create, workspace):
    secrets, _, session = accounts
    secrets.list.return_value = []
    with pytest.raises(RuntimeError):
        setup_accounts.check_accounts(create, expected_workspace=workspace)
    secrets.create.assert_not_called()
    session.assert_not_called()


def test_secret_copy_uses_same_environment_and_never_prints_values(accounts, capsys):
    secrets, _, _ = accounts
    secrets.list.return_value = []
    setup_accounts.check_accounts(True, expected_workspace='personal')
    secrets.create.assert_called_once_with('chefost-r2',
        {'AWS_ACCESS_KEY_ID': 'private-id', 'AWS_SECRET_ACCESS_KEY': 'private-key'}, environment_name='dev')
    output = capsys.readouterr().out
    assert 'private-id' not in output and 'private-key' not in output


def test_workspace_default_is_preserved(accounts):
    secrets, config, _ = accounts
    config.get.return_value = None
    setup_accounts.check_accounts()
    secrets.list.assert_called_once_with(environment_name='')
