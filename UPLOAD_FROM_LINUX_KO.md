# 리눅스에서 GitHub에 올리기

이 문서는 이 폴더 자체를 GitHub 저장소로 올리는 절차입니다. 원래 프로젝트 전체 폴더에서 `git add .`를 실행하지 마세요. 공개할 파일은 이 폴더에만 모아 두었습니다.

## 1. 전체 데이터 압축 준비

`Dataset`는 원래 프로젝트의 RGB/CTH/CTT 폴더를 포함하는 최상위 Dataset 경로입니다. 이미 압축이 끝나 `release_assets/SHA256SUMS`와 `data/dataset_files.sha256`가 있으면 이 단계는 건너뛸 수 있습니다.

```bash
cd /share/geosat-10/atmos_project/TCB_project/Code/Model/RGB/MambaVision/Revision_analysis/GitHub_release/TCB-Detection
python tools/package_dataset.py --source-root /share/geosat-10/atmos_project/TCB_project/Dataset
```

원본 NPY를 변경하지 않고 약 1.5 GiB 원자료 단위의 독립 ZIP으로 묶습니다. 압축파일은 `release_assets/`에 생성합니다. 중단되면 같은 명령으로 재개할 수 있습니다. GitHub 일반 저장소에는 이 대용량 ZIP을 넣지 않도록 `.gitignore`에 설정했습니다. 압축이 끝나면 `data/download_manifest.json`과 `data/dataset_files.sha256`도 완성됩니다.

## 2. 코드·설명·데이터 목록 push

아래 `YOUR_GITHUB_EMAIL`을 본인의 GitHub 등록 이메일로 바꿔서 실행하세요. 커밋 이름·이메일은 이 저장소에만 설정됩니다. 명령에서 오류가 나면 다음 단계로 넘어가지 말고 원인을 확인하세요.

```bash
cd /share/geosat-10/atmos_project/TCB_project/Code/Model/RGB/MambaVision/Revision_analysis/GitHub_release/TCB-Detection
git init
git config --global --add safe.directory "$PWD"
git remote add origin https://github.com/yangsy1509/TCB-Detection.git
git fetch origin main
git reset --mixed origin/main
git config user.name "yangsy1509"
git config user.email "YOUR_GITHUB_EMAIL"
git add .
git status --short
git commit -m "Release TCB dataset manifests, training and inference code"
git branch -M main
git push -u origin main
```

현재 GitHub 저장소에 README가 있으므로 `fetch`와 `reset --mixed`로 기존 커밋에 연결합니다. 이 명령은 준비한 파일의 내용을 바꾸지 않습니다. `--force`는 사용하지 않습니다. 오류가 나면 다음 명령을 계속 실행하지 말고 확인하세요.

`origin`이 이미 있으면 `git remote -v`로 위 주소인지 확인합니다. 다른 주소라면 `git remote set-url origin https://github.com/yangsy1509/TCB-Detection.git`으로 바꿉니다. 이미 `Author identity unknown` 오류가 났다면 재접속 후 이 폴더에서 위의 `git config user.name`, `git config user.email`을 설정하고 `git add .`부터 다시 진행하면 됩니다. `git init`, remote 설정, fetch/reset은 반복할 필요가 없습니다. 이전 안내의 `set -e`가 현재 셸에 남아 있다면 `set +e`로 해제할 수 있습니다. 토큰은 파일이나 코드에 적지 말고 Git 인증 프롬프트에서 입력합니다.

## 3. 데이터 ZIP을 GitHub Release에 업로드

웹에서 저장소의 Releases → 새 Release를 만들고 tag를 `v1.0.0`으로 지정합니다. `release_assets/`의 `tcb-dataset-part-*.zip` 전부와 `SHA256SUMS`를 첨부합니다. `package_state.json`은 로컬 압축 재개용이므로 첨부하지 않습니다. 각 ZIP은 2 GiB 미만이고 압축파일 전체가 같은 Release에 있어야 합니다.

GitHub CLI를 쓰는 경우, 먼저 `gh auth login`으로 yangsy1509 저장소에 쓰기 권한이 있는 계정으로 로그인한 다음:

```bash
bash tools/upload_release.sh
```

업로드 후 README의 다운로드 명령이 실행되는지 확인한 다음 논문에 저장소 주소를 넣습니다.

## 라이선스

모델 원본의 NVIDIA 라이선스·저작권 표기는 보존했습니다. 저자 코드와 데이터에 적용할 별도 라이선스는 아직 지정하지 않았습니다. 공개 사용 조건을 정하면 LICENSE와 데이터 사용 조건을 추가하세요.
