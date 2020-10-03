{
    "variables": {
        "provider": null,
        "version": null,
        "access_token": null,
        "skip_add": false
    },
    "builders": [
        {
            "type": "vagrant",
            "communicator": "ssh",
            "source_path": "generic/ubuntu2004",
            "provider": "virtualbox",
            "template": "Vagrantfile.packer-template",
            "skip_add": "{{user `skip_add`}}"
        }
    ],
    "post-processors": [
        {
            "type": "vagrant-cloud",
            "box_tag": "areusch/microtvm-staging",
            "version": "{{user `version`}}",
            "access_token": "{{user `access_token`}}"
        }
    ]
}
