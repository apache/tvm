---
name: "\U0001F40B Update CI Docker Image"
about: Provide information on CI Docker Images requiring updates
title: "[CI] "

---

Thanks for participating in the TVM community! We use https://discuss.tvm.ai for any general usage questions and discussions. The issue tracker is used for actionable items such as feature proposals discussion, roadmaps, and bug tracking.  You are always welcomed to post on the forum first :smile_cat:

Issues that are inactive for a period of time may get closed. We adopt this policy so that we won't lose track of actionable issues that may fall at the bottom of the pile. Feel free to reopen a new one if you feel there is an additional problem that needs attention when an old one gets closed.

- [ ] S0. Reason: For example, a blocked PR or a feature issue

- [ ] S1. Tag of nightly build: TAG. Docker hub: https://hub.docker.com/layers/tlcpackstaging/ci_cpu/...

- [ ] S2. The nightly is built on TVM commit: TVM_COMMIT. Detailed info can be found here: https://ci.tlcpack.ai/blue/organizations/jenkins/docker-images-ci%2Fdaily-docker-image-rebuild/detail/daily-docker-image-rebuild/....

- [ ] S3. Testing the nightly image on ci-docker-staging: https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/detail/ci-docker-staging/...

- [ ] S4. Retag TAG to VERSION:
```
docker pull tlcpackstaging/IMAGE_NAME:TAG
docker tag tlcpackstaging/IMAGE_NAME:TAG tlcpack/IMAGE_NAME:VERSION
docker push tlcpack/IMAGE_NAME:VERSION
```

- [ ] S5. Check if the new tag is really there: https://hub.docker.com/u/tlcpack

- [ ] S6. Submit a PR updating the IMAGE_NAME version on Jenkins
